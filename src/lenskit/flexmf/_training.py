# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import platform
from abc import abstractmethod
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np
import structlog
import torch
from torch import Tensor
from typing_extensions import TYPE_CHECKING, Generic, Self, TypeVar

from lenskit.data import Dataset, MatrixRelationshipSet
from lenskit.logging import get_logger, item_progress
from lenskit.parallel.config import ensure_parallel_init
from lenskit.training import ModelTrainer, TrainingOptions

from ._model import FlexMFModel

try:
    from torch._dynamo.exc import TorchDynamoException
except ImportError:
    TorchDynamoException = None

# hide base import to avoid circular imports
if TYPE_CHECKING:
    from ._base import FlexMFConfigBase, FlexMFScorerBase


Comp = TypeVar("Comp", bound="FlexMFScorerBase", default="FlexMFScorerBase")
Cfg = TypeVar("Cfg", bound="FlexMFConfigBase", default="FlexMFConfigBase")

_log = get_logger(__name__)


class FlexMFTrainerBase(ModelTrainer, Generic[Comp, Cfg]):
    """
    Trainer for a FlexMF model.  This class should be inherited by the
    model-specific trainers.
    """

    component: Comp
    """
    The component whose model is being trained.
    """
    opt: torch.optim.Optimizer
    """
    The PyTorch optimizer.
    """
    data: FlexMFTrainingData
    """
    The training data, set up for the training process.
    """
    epochs_trained: int = 0
    """
    The number of epochs trained so far.
    """

    device: str
    """
    Torch device for training.
    """

    explicit_norm: bool
    """
    Use explicit (L2) norm for normalizing data.
    """

    rng: np.random.Generator
    """
    NumPy generator for random number generation.
    """
    torch_rng: torch.Generator
    """
    PyPTorch generator for random number generation.
    """

    log: structlog.stdlib.BoundLogger = field(default_factory=lambda: _log.bind())
    """
    A logger, that is bound the current training status / position.
    """

    _compiled_model: Callable[..., torch.Tensor] | None = None

    def __init__(self, component: Comp, data: Dataset, options: TrainingOptions):
        ensure_parallel_init()

        self.component = component
        self.explicit_norm = component.config.reg_method == "L2"

        self.log = _log.bind(scorer=self.__class__.__name__, size=self.config.embedding_size)

        self.device = options.configured_device(gpu_default=True)
        self._init_rng(options)

        self.data = self.prepare_data(data)
        self.component.model = self.create_model()

        # zero out non-interacted users/items
        users = data.user_stats()
        self.model.zero_users(torch.tensor(users["count"].values == 0))
        items = data.item_stats()
        self.model.zero_items(torch.tensor(items["count"].values == 0))

        self.log.info("preparing to train %r", self.component, device=self.device)
        self.component.model = self.model.to(self.device)
        self.model.train(True)

        if platform.system() not in ("Linux", "Darwin"):
            _log.warn("compiled models are only usable on Linux and macOS")
        elif torch.__version__ < "2.8":
            _log.warn("compiled models require Torch >=2.8")
        else:
            _log.debug("compiling FlexMF model")
            self._compiled_model = torch.compile(self.model)

        self.setup_optimizer()

    @property
    def config(self) -> Cfg:
        """
        Get the component configuration.
        """
        return self.component.config  # type: ignore

    @property
    def model(self) -> FlexMFModel:
        """
        Get model being trained.
        """
        return self.component.model

    def call_model(self, *args, **kwargs) -> torch.Tensor:
        """
        Invoke the model, using the compiled version if available.
        """
        if self._compiled_model is not None:
            assert TorchDynamoException is not None
            try:
                return self._compiled_model(*args, **kwargs)
            except TorchDynamoException as e:
                self.log.warning("calling compiled model failed, falling back: %s", e)
                self._compiled_model = None
                return self.call_model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)

    def train_epoch(self) -> dict[str, float]:
        epoch = self.epochs_trained + 1
        self.log = elog = self.log.bind(epoch=epoch)
        elog.debug("creating epoch training data")
        epoch_data = self.epoch_data()

        tot_loss = torch.tensor(0.0).to(self.device)
        avg_loss = np.nan
        with item_progress(
            f"Training epoch {epoch}", epoch_data.batch_count, {"loss": ".3f"}
        ) as pb:
            elog.debug("beginning epoch")
            for i, batch in enumerate(epoch_data.batches(), 1):
                self.log = blog = elog.bind(batch=i)
                blog.debug("training batch")
                loss = self.train_batch(batch)
                self.opt.zero_grad()

                if i % 20 == 0:
                    avg_loss = tot_loss.item() / epoch_data.batch_count
                pb.update(loss=avg_loss)
                tot_loss += loss

        avg_loss = tot_loss.item() / epoch_data.batch_count
        elog.debug("epoch complete", loss=avg_loss)
        self.epochs_trained += 1
        return {"loss": avg_loss}

    @abstractmethod
    def train_batch(self, batch: FlexMFTrainingBatch) -> float:  # pragma: nocover
        """
        Compute and apply updates for a single batch.

        Args:
            batch:
                The training minibatch.

        Returns:
            The loss.
        """

    def finalize(self):
        del self.opt
        del self.data
        self.model.eval()

    def epoch_data(self) -> FlexMFTrainingEpoch:
        """
        Get training data for a single training epoch.
        """
        # permute the data
        perm = np.require(self.rng.permutation(self.data.n_samples), dtype=np.int32)
        # convert to tensor, send to the training data's device.
        if isinstance(self.data.items, torch.Tensor):
            perm = torch.tensor(perm).to(self.data.items.device)

        return FlexMFTrainingEpoch(self.data, perm)

    @abstractmethod
    def prepare_data(self, data: Dataset) -> FlexMFTrainingData:  # pragma: nocover
        """
        Set up the training data and context for the scorer.

        This method _returns_ the data object; the :meth:`setup` method will
        save its return value in :attr:`data`.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_model(self) -> FlexMFModel:  # pragma: nocover
        """
        Create and initialize the model for training.

        This method should _return_ the model; the :meth:`setup` method will put
        the model in the component and arrange for it to be available via
        :attr:`model`.
        """
        raise NotImplementedError()

    def setup_optimizer(self):
        """
        Create the appropriate optimizer depending on the regularization method.
        """
        if self.config.reg_method == "AdamW":
            self.log.debug("creating AdamW optimizer")
            opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.regularization,
            )
        else:
            self.log.debug("creating SparseAdam optimizer")
            opt = torch.optim.SparseAdam(self.model.parameters(), lr=self.config.learning_rate)
        self.opt = opt

    def _init_rng(self, options: TrainingOptions):
        self.rng = options.random_generator()

        # use the NumPy generator to seed Torch
        self.torch_rng = torch.Generator()
        i32 = np.iinfo(np.int32)
        self.torch_rng.manual_seed(int(self.rng.integers(i32.min, i32.max)))

    def get_parameters(self) -> Mapping[str, object]:
        return self.model.state_dict()

    def load_parameters(self, state: Mapping[str, object]) -> None:
        self.model.load_state_dict(state)


@dataclass
class FlexMFTrainingData:
    """
    Training data for a FlexMF model.

    Stability:
        Experimental
    """

    batch_size: int

    n_users: int
    n_items: int

    users: torch.Tensor | np.ndarray[tuple[int], np.dtype[np.int32]]
    "User numbers for training samples."
    items: torch.Tensor | np.ndarray[tuple[int], np.dtype[np.int32]]
    "Item numbers for training samples."

    matrix: MatrixRelationshipSet | None = None
    """
    The original relationship set we are training on.
    """

    fields: dict[str, torch.Tensor] = field(default_factory=dict)
    "Additional per-sample data fields."

    def to(self, device: str) -> Self:
        """
        Move this data to another device.
        """
        ut = torch.as_tensor(self.users, device=device)
        it = torch.as_tensor(self.items, device=device)
        fts = {f: torch.as_tensor(t, device=device) for (f, t) in self.fields.items()}
        return replace(self, users=ut, items=it, fields=fts)

    @property
    def n_samples(self) -> int:
        return len(self.users)


@dataclass
class FlexMFTrainingEpoch:
    """
    Permuted data for a single training epoch.
    """

    data: FlexMFTrainingData
    permutation: torch.Tensor | np.ndarray[tuple[int], np.dtype[np.int32]]

    @property
    def n_samples(self) -> int:
        return self.data.n_samples

    @property
    def batch_size(self) -> int:
        return self.data.batch_size

    @property
    def batch_count(self) -> int:
        return math.ceil(self.n_samples / self.batch_size)

    def batches(
        self, fields: Sequence[str] | set[str] | None = None
    ) -> Generator[FlexMFTrainingBatch, None, None]:
        """
        Iterate over batches in this epoch.

        Args:
            fields:
                The fields to include in the batch.
        """

        fields = set(self.data.fields.keys()) if fields is None else fields
        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            rows = self.permutation[start:end]

            yield self.make_batch(rows, fields)

    def make_batch(
        self,
        rows: Tensor | np.ndarray[tuple[int], np.dtype[np.int32]],
        fields: Sequence[str] | set[str],
    ) -> FlexMFTrainingBatch:
        ut = self.data.users[rows]
        it = self.data.items[rows]
        fts = {f: self.data.fields[f][rows] for f in fields}
        return FlexMFTrainingBatch(self.data, ut, it, fts)


@dataclass
class FlexMFTrainingBatch:
    "Representation of a single batch."

    data: FlexMFTrainingData
    users: torch.Tensor | np.ndarray[tuple[int], np.dtype[np.int32]]
    items: torch.Tensor | np.ndarray[tuple[int], np.dtype[np.int32]]
    fields: dict[str, torch.Tensor]

    def to(self, device: str) -> Self:
        """
        Move this data to another device.
        """
        ut = torch.as_tensor(self.users, device=device)
        it = torch.as_tensor(self.items, device=device)
        fts = {f: torch.as_tensor(t, device=device) for (f, t) in self.fields.items()}
        return replace(self, users=ut, items=it, fields=fts)
