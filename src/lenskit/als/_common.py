# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Mapping, TypeAlias, cast

import numpy as np
import structlog
from pydantic import AliasChoices, BaseModel, Field, PositiveFloat, PositiveInt
from scipy.sparse import coo_array
from typing_extensions import Generic, NamedTuple, TypeVar, override

from lenskit.config.common import EmbeddingSizeMixin
from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.data.matrix import SparseRowArray
from lenskit.data.types import NPMatrix, NPVector, UIPair
from lenskit.logging import get_logger
from lenskit.parallel import ensure_parallel_init
from lenskit.pipeline import Component
from lenskit.training import ModelTrainer, TrainingOptions, UsesTrainer

_log = get_logger(__name__)

EntityClass: TypeAlias = Literal["user", "item"]

Scorer = TypeVar("Scorer", bound="ALSBase")
Config = TypeVar("Config", bound="ALSConfig")


class ALSConfig(EmbeddingSizeMixin, BaseModel):
    """
    Configuration for ALS scorers.
    """

    embedding_size: PositiveInt = Field(
        default=50, validation_alias=AliasChoices("embedding_size", "features")
    )
    """
    The dimension of user and item embeddings (number of latent features to
    learn).
    """
    epochs: PositiveInt = 10
    """
    The number of epochs to train.
    """
    regularization: PositiveFloat | UIPair[PositiveFloat] = 0.1
    """
    L2 regularization strength.
    """
    user_embeddings: bool | Literal["prefer"] = True
    """
    Whether to retain user embeddings after training.  If ``True``, they are
    retained, but are ignored if the query has historical items; if ``False``,
    they are not. If set to ``"prefer"``, then the user embeddings from training
    time are used even if the query has a user history.  This makes inference
    faster when histories only consist of the user's items from the training
    set.
    """

    @property
    def user_reg(self) -> float:
        if isinstance(self.regularization, UIPair):
            return self.regularization.user
        else:
            return self.regularization

    @property
    def item_reg(self) -> float:
        if isinstance(self.regularization, UIPair):
            return self.regularization.item
        else:
            return self.regularization


class TrainContext(NamedTuple):
    """
    Context object for one half of an ALS training operation.
    """

    label: str
    matrix: SparseRowArray
    left: NPMatrix
    right: NPMatrix
    reg: float
    nrows: int
    ncols: int
    embed_size: int
    regI: NPMatrix

    @classmethod
    def create(
        cls,
        label: str,
        matrix: SparseRowArray,
        left: NPMatrix,
        right: NPMatrix,
        reg: float,
    ) -> TrainContext:
        nrows, ncols = matrix.shape
        lnr, embed_size = left.shape
        assert lnr == nrows
        assert right.shape == (ncols, embed_size)
        regI = np.eye(embed_size, dtype=left.dtype) * reg
        return TrainContext(label, matrix, left, right, reg, nrows, ncols, embed_size, regI)


class ALSBase(UsesTrainer, Component[ItemList], ABC):
    """
    Base class for ALS models.

    Stability:
        Caller
    """

    config: ALSConfig

    users: Vocabulary | None
    items: Vocabulary
    user_embeddings: NPMatrix | None
    item_embeddings: NPMatrix

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        return _log.bind(scorer=self.__class__.__name__, size=self.config.embedding_size)

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        user_id = query.user_id
        user_num = None
        if user_id is not None and self.users is not None:
            user_num = self.users.number(user_id, missing=None)

        log = self.logger.bind(user=user_id)

        u_offset = None
        u_feat = None
        if (
            query.user_items is not None
            and len(query.user_items) > 0
            and self.config.user_embeddings != "prefer"
        ):
            log.debug("training user embedding")
            u_feat, u_offset = self.new_user_embedding(user_num, query.user_items)

        if u_feat is None:
            if user_num is None or self.user_embeddings is None:
                log.debug("cannot find user embedding")
                return ItemList(items, scores=np.nan)
            u_feat = self.user_embeddings[user_num, :]

        item_nums = items.numbers(vocabulary=self.items, missing="negative")
        item_mask = item_nums >= 0

        scores = np.full((len(items),), np.nan, dtype=np.float32)
        if len(item_nums) <= self.item_embeddings.shape[0] * 0.8:
            # small set — subset inputs
            i_feats = self.item_embeddings[item_nums[item_mask], :]
            scores[item_mask] = i_feats @ u_feat
        else:
            # larger set — subset outputs
            allmult = self.item_embeddings @ u_feat
            scores[item_mask] = allmult[item_nums[item_mask]]

        log.debug("scored %d items", np.sum(item_mask))

        results = ItemList(items, scores=scores)
        return self.finalize_scores(user_num, results, u_offset)

    @abstractmethod
    def new_user_embedding(
        self, user_num: int | None, items: ItemList
    ) -> tuple[NPVector[np.float32], float | None]:  # pragma: no cover
        """
        Generate an embedding for a user given their current ratings.
        """
        ...

    def finalize_scores(
        self, user_num: int | None, items: ItemList, user_bias: float | None
    ) -> ItemList:
        """
        Perform any final transformation of scores prior to returning them.
        """
        return items


class ALSTrainerBase(ModelTrainer, Generic[Scorer, Config]):
    scorer: Scorer

    rng: np.random.Generator
    ui_rates: SparseRowArray
    "User-item rating matrix."
    u_ctx: TrainContext
    "User training context."
    iu_rates: SparseRowArray
    "Item-user rating matrix."
    i_ctx: TrainContext
    "Item training context."
    epochs_trained: int = 0

    def __init__(self, scorer: Scorer, data: Dataset, options: TrainingOptions):
        ensure_parallel_init()
        self.scorer = scorer
        self.scorer.users = data.users
        self.scorer.items = data.items

        self.rng = options.random_generator()

        ui_rates = self.prepare_matrix(data)
        self.ui_rates = SparseRowArray.from_scipy(ui_rates)
        self.iu_rates = SparseRowArray.from_scipy(ui_rates.T)

        self.initialize_params(data)
        self._init_contexts()

    def _init_contexts(self):
        assert self.scorer.user_embeddings is not None
        self.u_ctx = TrainContext.create(
            "user",
            self.ui_rates,
            self.scorer.user_embeddings,
            self.scorer.item_embeddings,
            self.config.user_reg,
        )
        self.i_ctx = TrainContext.create(
            "item",
            self.iu_rates,
            self.scorer.item_embeddings,
            self.scorer.user_embeddings,
            self.config.item_reg,
        )

    def train_epoch(self):
        epoch = self.epochs_trained + 1
        log = self.logger.bind(epoch=epoch)

        assert self.scorer.user_embeddings is not None
        assert self.scorer.item_embeddings is not None

        du = self.als_half_epoch(epoch, self.u_ctx)
        log.debug("finished user epoch")

        di = self.als_half_epoch(epoch, self.i_ctx)
        log.debug("finished item epoch")

        log.debug("finished epoch (|ΔP|=%.3f, |ΔQ|=%.3f)", du, di)
        self.epochs_trained += 1
        return {"deltaP": du, "deltaQ": di}

    @property
    def config(self) -> Config:
        return cast(Config, self.scorer.config)

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        return self.scorer.logger

    @property
    def n_users(self):
        return self.ui_rates.shape[0]

    @property
    def n_items(self):
        return self.iu_rates.shape[1]

    @abstractmethod
    def prepare_matrix(self, data: Dataset) -> coo_array:  # pragma: no cover
        """
        Prepare data for training this model.  This takes in the ratings, and is
        supposed to do two things:

        -   Normalize or transform the rating/interaction data, as needed, for
            training.
        -   Store any parameters learned from the normalization (e.g. means) in
            the appropriate member variables.
        -   Return the ratings matrix for training.
        """

    def initialize_params(self, data: Dataset):
        """
        Initialize the model parameters at the beginning of training.
        """
        self.logger.debug("initializing item matrix")
        self.scorer.item_embeddings = self.initial_params(
            data.item_count, self.config.embedding_size
        )
        self.logger.debug("|Q|: %f", np.linalg.norm(self.scorer.item_embeddings, "fro"))

        self.logger.debug("initializing user matrix")
        self.scorer.user_embeddings = self.initial_params(
            data.user_count, self.config.embedding_size
        )
        self.logger.debug("|P|: %f", np.linalg.norm(self.scorer.user_embeddings, "fro"))

    @abstractmethod
    def initial_params(self, nrows: int, ncols: int) -> NPMatrix:  # pragma: no cover
        """
        Compute initial parameter values of the specified shape.
        """
        ...

    @abstractmethod
    def als_half_epoch(self, epoch: int, context: TrainContext) -> float:  # pragma: no cover
        """
        Run one half of an ALS training epoch.
        """
        ...

    @override
    def finalize(self):
        """
        Finalize training. Base classes must call superclass.
        """
        self.logger.debug("finalizing model training")
        if not self.config.user_embeddings:
            self.scorer.user_embeddings = None
            self.scorer.users = None

    def get_parameters(self) -> Mapping[str, object]:
        """
        Get the component's parameters.

        Returns:
            The model's parameters, as a dictionary from names to parameter data
            (usually arrays, tensors, etc.).
        """
        return {
            "user_embeddings": self.scorer.user_embeddings,
            "item_embeddings": self.scorer.item_embeddings,
        }

    def load_parameters(self, state: Mapping[str, object]) -> None:
        """
        Reload model state from parameters saved via :meth:`get_parameters`.

        Args:
            params:
                The model parameters, as a dictionary from names to parameter
                data (arrays, tensors, etc.), as returned from
                :meth:`get_parameters`.
        """
        u_emb = state["user_embeddings"]
        assert isinstance(u_emb, np.ndarray)
        i_emb = state["item_embeddings"]
        assert isinstance(i_emb, np.ndarray)
        self.scorer.user_embeddings = u_emb
        self.scorer.item_embeddings = i_emb
        self._init_contexts()
