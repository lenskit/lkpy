# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Mapping, TypeAlias

import numpy as np
import structlog
import torch
from pydantic import AliasChoices, BaseModel, Field
from typing_extensions import NamedTuple, override

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.data.types import UIPair
from lenskit.logging import get_logger
from lenskit.pipeline import Component
from lenskit.training import ModelTrainer, TrainingOptions, UsesTrainer

EntityClass: TypeAlias = Literal["user", "item"]
_log = get_logger(__name__)


class ALSConfig(BaseModel):
    """
    Configuration for ALS scorers.
    """

    embedding_size: int = Field(
        default=50, validation_alias=AliasChoices("embedding_size", "features")
    )
    """
    The dimension of user and item embeddings (number of latent features to
    learn).
    """
    epochs: int = 10
    """
    The number of epochs to train.
    """
    regularization: float | UIPair[float] = 0.1
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
    matrix: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    reg: float
    nrows: int
    ncols: int
    embed_size: int
    regI: torch.Tensor

    @classmethod
    def create(
        cls, label: str, matrix: torch.Tensor, left: torch.Tensor, right: torch.Tensor, reg: float
    ) -> TrainContext:
        nrows, ncols = matrix.shape
        lnr, embed_size = left.shape
        assert lnr == nrows
        assert right.shape == (ncols, embed_size)
        regI = torch.eye(embed_size, dtype=left.dtype, device=left.device) * reg
        return TrainContext(label, matrix, left, right, reg, nrows, ncols, embed_size, regI)


class ALSBase(UsesTrainer, Component[ItemList], ABC):
    """
    Base class for ALS models.

    Stability:
        Caller
    """

    config: ALSConfig

    users_: Vocabulary | None
    items_: Vocabulary
    user_features_: torch.Tensor | None
    item_features_: torch.Tensor

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        return _log.bind(scorer=self.__class__.__name__, size=self.config.embedding_size)

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        user_id = query.user_id
        user_num = None
        if user_id is not None and self.users_ is not None:
            user_num = self.users_.number(user_id, missing=None)

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
            if user_num is None or self.user_features_ is None:
                log.debug("cannot find user embedding")
                return ItemList(items, scores=np.nan)
            u_feat = self.user_features_[user_num, :]

        item_nums = items.numbers("torch", vocabulary=self.items_, missing="negative")
        item_mask = item_nums >= 0
        i_feats = self.item_features_[item_nums[item_mask], :]

        scores = torch.full((len(items),), np.nan, dtype=torch.float64)
        scores[item_mask] = i_feats @ u_feat
        log.debug("scored %d items", torch.sum(item_mask).item())

        results = ItemList(items, scores=scores)
        return self.finalize_scores(user_num, results, u_offset)

    @abstractmethod
    def new_user_embedding(
        self, user_num: int | None, items: ItemList
    ) -> tuple[torch.Tensor, float | None]:  # pragma: no cover
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


class ALSTrainerBase(ModelTrainer):
    scorer: ALSBase

    rng: np.random.Generator
    ui_rates: torch.Tensor
    "User-item rating matrix."
    u_ctx: TrainContext
    iu_rates: torch.Tensor
    "Item-user rating matrix."
    i_ctx: TrainContext
    epochs_trained: int = 0

    def __init__(self, scorer: ALSBase, data: Dataset, options: TrainingOptions):
        self.scorer = scorer
        self.scorer.users_ = data.users
        self.scorer.items_ = data.items

        self.rng = options.random_generator()

        self.ui_rates = self.prepare_matrix(data)
        self.iu_rates = self.ui_rates.transpose(0, 1).to_sparse_csr()

        self.initialize_params(data)
        self._init_contexts()

    def _init_contexts(self):
        assert self.scorer.user_features_ is not None
        self.u_ctx = TrainContext.create(
            "user",
            self.ui_rates,
            self.scorer.user_features_,
            self.scorer.item_features_,
            self.config.user_reg,
        )
        self.i_ctx = TrainContext.create(
            "item",
            self.iu_rates,
            self.scorer.item_features_,
            self.scorer.user_features_,
            self.config.item_reg,
        )

    def train_epoch(self):
        epoch = self.epochs_trained + 1
        log = self.logger.bind(epoch=epoch)

        assert self.scorer.user_features_ is not None
        assert self.scorer.item_features_ is not None

        du = self.als_half_epoch(epoch, self.u_ctx)
        log.debug("finished user epoch")

        di = self.als_half_epoch(epoch, self.i_ctx)
        log.debug("finished item epoch")

        log.debug("finished epoch (|ΔP|=%.3f, |ΔQ|=%.3f)", du, di)
        return {"deltaP": du, "deltaQ": di}

    @property
    def config(self) -> ALSConfig:
        return self.scorer.config

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
    def prepare_matrix(self, data: Dataset) -> torch.Tensor:  # pragma: no cover
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
        self.scorer.item_features_ = self.initial_params(
            data.item_count, self.config.embedding_size
        )
        self.logger.debug("|Q|: %f", torch.norm(self.scorer.item_features_, "fro"))

        self.logger.debug("initializing user matrix")
        self.scorer.user_features_ = self.initial_params(
            data.user_count, self.config.embedding_size
        )
        self.logger.debug("|P|: %f", torch.norm(self.scorer.user_features_, "fro"))

    @abstractmethod
    def initial_params(self, nrows: int, ncols: int) -> torch.Tensor:  # pragma: no cover
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
            self.scorer.user_features_ = None
            self.scorer.users_ = None

    def get_parameters(self) -> Mapping[str, object]:
        """
        Get the component's parameters.

        Returns:
            The model's parameters, as a dictionary from names to parameter data
            (usually arrays, tensors, etc.).
        """
        return {
            "user_embeddings": self.scorer.user_features_,
            "item_embeddings": self.scorer.item_features_,
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
        assert torch.is_tensor(u_emb)
        i_emb = state["item_embeddings"]
        assert torch.is_tensor(i_emb)
        self.scorer.user_features_ = u_emb
        self.scorer.item_features_ = i_emb
        self._init_contexts()
