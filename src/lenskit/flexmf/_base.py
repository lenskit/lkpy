# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.logging import get_logger, item_progress
from lenskit.parallel.config import ensure_parallel_init
from lenskit.pipeline import Component
from lenskit.training import IterativeTraining, TrainingOptions

from ._model import FlexMFModel
from ._training import FlexMFTrainingBatch, FlexMFTrainingContext, FlexMFTrainingData

# I want a logger for information
_log = get_logger(__name__)


@dataclass
class FlexMFConfigBase:
    """
    Common configuration for all FlexMF scoring components.

    Stability:
        Experimental
    """

    embedding_size: int = 50
    """
    The dimension of the embedding space (number of latent features).
    """

    batch_size: int = 8 * 1024
    """
    The training batch size.
    """

    learning_rate: float = 0.005
    """
    The learning rate for training.
    """

    epochs: int = 20
    """
    The number of training epochs.
    """

    reg: float = 0.1
    """
    The regularization strength.
    """

    reg_method: Literal["AdamW", "L2"] | None = "L2"
    """
    The regularization method to use.

    With the default L2 regularization, training will use sparse gradients and
    the :class:`torch.optim.SparseAdam` optimizer.

    ``None``
        Use no regularization.

    ``"L2"``
        Use L2 regularization on the parameters used in each training batch.
        The strength is applied to the _mean_ norms in a batch, so that the
        regularization term scale is not dependent on the batch size.

    ``"AdamW"``
        Use :class:`torch.optim.AdamW` with the specified regularization
        strength.  This configuration does *not* use sparse gradients and may
        train more slowly.

    .. note::
        Regularization values do not necessarily have the same range or meaning
        for the different regularization methods.
    """


class FlexMFScorerBase(IterativeTraining, Component):
    """
    Base class for the FlexMF scorers, providing common Torch support.

    Stability:
        Experimental
    """

    config: FlexMFConfigBase
    users: Vocabulary
    items: Vocabulary
    model: FlexMFModel

    def training_loop(
        self, data: Dataset, options: TrainingOptions
    ) -> Generator[dict[str, float], None, None]:
        ensure_parallel_init()
        train_ctx = self.prepare_context(options)
        train_data = self.prepare_data(data, options, train_ctx)
        self.model = self.create_model(train_ctx, train_data)

        # zero out non-interacted users/items
        users = data.user_stats()
        self.model.zero_users(torch.tensor(users["count"].values == 0))
        items = data.item_stats()
        self.model.zero_items(torch.tensor(items["count"].values == 0))

        _log.info("preparing to train", device=train_ctx.device, model=self)
        self.model = self.model.to(train_ctx.device)
        self.model.train(True)

        # delegate to the inner training loop
        return self._training_loop_impl(train_data, train_ctx)

    def prepare_context(self, options: TrainingOptions) -> FlexMFTrainingContext:
        device = options.configured_device(gpu_default=True)
        rng = options.random_generator()

        # use the NumPy generator to seed Torch
        torch_rng = torch.Generator()
        i32 = np.iinfo(np.int32)
        torch_rng.manual_seed(int(rng.integers(i32.min, i32.max)))

        return FlexMFTrainingContext(device, rng, torch_rng)

    @abstractmethod
    def prepare_data(
        self, data: Dataset, options: TrainingOptions, context: FlexMFTrainingContext
    ) -> FlexMFTrainingData:  # pragma: nocover
        """
        Set up the training data and context for the scorer.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_model(
        self, context: FlexMFTrainingContext, data: FlexMFTrainingData
    ) -> FlexMFModel:  # pragma: nocover
        """
        Prepare the model for training.
        """
        raise NotImplementedError()

    def create_optimizer(self, context: FlexMFTrainingContext) -> torch.optim.Optimizer:
        """
        Create the appropriate optimizer depending on the regularization method.
        """
        if self.config.reg_method == "AdamW":
            context.log.debug("creating AdamW optimizer")
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.reg,
            )
        else:
            context.log.debug("creating SparseAdam optimizer")
            return torch.optim.SparseAdam(self.model.parameters(), lr=self.config.learning_rate)

    def _training_loop_impl(self, data: FlexMFTrainingData, context: FlexMFTrainingContext):
        log = _log.bind(model=self.__class__.__name__, size=self.config.embedding_size)
        context.log = log
        opt = self.create_optimizer(context)

        for epoch in range(1, self.config.epochs + 1):
            # permute and copy the training data
            epoch_data = data.epoch(context)
            context.log = elog = log.bind(epoch=epoch)

            tot_loss = 0.0
            with item_progress(
                f"Training epoch {epoch}", epoch_data.batch_count, {"loss": ".3f"}
            ) as pb:
                for i, batch in enumerate(epoch_data.batches(), 1):
                    context.log = blog = elog.bind(batch=i)
                    blog.debug("training batch")
                    opt.zero_grad()
                    loss = self.train_batch(context, batch, opt)

                    pb.update(loss=loss)
                    tot_loss += loss

            yield {"loss": tot_loss / epoch_data.batch_count}

        _log.info("finalizing trained model")
        self.finalize()

    @abstractmethod
    def train_batch(
        self, context: FlexMFTrainingContext, batch: FlexMFTrainingBatch, opt: torch.optim.Optimizer
    ) -> float:  # pragma: nocover
        """
        Compute and apply updates for a single batch.

        Args:
            batch:
                The training minibatch.
            opt:
                The optimizer (its gradients have already been zeroed).

        Returns:
            The loss.
        """
        raise NotImplementedError()

    def finalize(self):
        """
        Finalize model training.  The base class implementation puts the model
        in evaluation mode.
        """
        self.model.eval()

    def to(self, device):
        "Move the model to a different device."
        self.model = self.model.to(device)
        return self

    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        """
        Generate item scores for a user.

        Note that user and items are both user and item IDs, not positions.
        """
        # make sure the query is in a known / usable format
        query = RecQuery.create(query)

        # if we have no user ID, we cannot score items
        # TODO: support pooling from user history
        u_row = None
        if query.user_id is not None:
            u_row = self.users.number(query.user_id, missing=None)

        if u_row is None:
            return ItemList(items, scores=np.nan)

        # look up the user row in the embedding matrix
        u_tensor = torch.IntTensor([u_row])
        # make sure it's on the right device
        device = self.model.device
        u_tensor = u_tensor.to(device)

        # look up the item columns in the embedding matrix
        i_cols = items.numbers(vocabulary=self.items, missing="negative")

        # unknown items will have column -1 - limit to the
        # ones we know, and remember which item IDs those are
        scorable_mask = i_cols >= 0
        i_cols = i_cols[scorable_mask]
        i_tensor = torch.from_numpy(i_cols)
        i_tensor = i_tensor.to(device)

        # initialize output score array, fill with missing
        full_scores = np.full(len(items), np.nan, dtype=np.float32)

        # get scores
        with torch.inference_mode():
            scores = self.score_items(u_tensor, i_tensor)

        # fill in scores for scorable items
        full_scores[scorable_mask] = scores.cpu()

        # return the result!
        return ItemList(items, scores=full_scores)

    def score_items(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Score for users and items, after resolivng them and limiting to known
        users and items.
        """
        return self.model(users, items)
