# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from progress_api import make_progress
from typing_extensions import Iterator, NamedTuple, Self, override

from lenskit import util
from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.data.types import UITuple
from lenskit.parallel.config import ensure_parallel_init
from lenskit.pipeline import Component, Trainable
from lenskit.types import RNGInput
from lenskit.util.random import random_generator


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


class TrainingData(NamedTuple):
    """
    Data for training the ALS model.
    """

    users: Vocabulary
    "User ID mapping."
    items: Vocabulary
    "Item ID mapping."
    ui_rates: torch.Tensor
    "User-item rating matrix."
    iu_rates: torch.Tensor
    "Item-user rating matrix."

    @property
    def n_users(self):
        return len(self.users)

    @property
    def n_items(self):
        return len(self.items)

    @classmethod
    def create(cls, users: Vocabulary, items: Vocabulary, ratings: torch.Tensor) -> TrainingData:
        assert ratings.shape == (len(users), len(items))

        transposed = ratings.transpose(0, 1).to_sparse_csr()
        return cls(users, items, ratings, transposed)

    def to(self, device):
        """
        Move the training data to another device.
        """
        return self._replace(ui_rates=self.ui_rates.to(device), iu_rates=self.iu_rates.to(device))


class ALSBase(ABC, Component, Trainable):
    """
    Base class for ALS models.
    """

    features: int
    epochs: int
    reg: UITuple[float]
    rng: RNGInput
    save_user_features: bool

    users_: Vocabulary | None
    items_: Vocabulary
    user_features_: torch.Tensor | None
    item_features_: torch.Tensor

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:  # pragma: no cover
        """
        Overridden in implementation to provide the logger.
        """
        ...

    def __init__(
        self,
        features: int,
        *,
        epochs: int = 10,
        reg: UITuple[float] | float | tuple[float, float] = 0.1,
        save_user_features: bool = True,
        rng: RNGInput = None,
    ):
        self.features = features
        self.epochs = epochs
        self.reg = UITuple.create(reg)
        self.rng = rng
        self.save_user_features = save_user_features

    @property
    def is_trained(self) -> bool:
        return hasattr(self, "item_features_")

    @override
    def train(self, data: Dataset):
        """
        Run ALS to train a model.

        Args:
            ratings: the ratings data frame.
        """
        ensure_parallel_init()
        timer = util.Stopwatch()

        for algo in self.fit_iters(data, timer=timer):
            pass  # we just need to do the iterations

        if self.user_features_ is not None:
            self.logger.info(
                "trained model in %s (|P|=%f, |Q|=%f)",
                timer,
                torch.norm(self.user_features_, "fro"),
                torch.norm(self.item_features_, "fro"),
            )
        else:
            self.logger.info(
                "trained model in %s (|Q|=%f)",
                timer,
                torch.norm(self.item_features_, "fro"),
            )

    def fit_iters(self, data: Dataset, *, timer: util.Stopwatch | None = None) -> Iterator[Self]:
        """
        Run ALS to train a model, yielding after each iteration.

        Args:
            ratings: the ratings data frame.
        """
        if timer is None:
            timer = util.Stopwatch()

        train = self.prepare_data(data)
        self.users_ = train.users
        self.items_ = train.items

        self.initialize_params(train)

        if isinstance(self.reg, tuple):
            ureg, ireg = self.reg
        else:
            ureg = ireg = self.reg

        assert self.user_features_ is not None
        assert self.item_features_ is not None
        u_ctx = TrainContext.create(
            "user", train.ui_rates, self.user_features_, self.item_features_, ureg
        )
        i_ctx = TrainContext.create(
            "item", train.iu_rates, self.item_features_, self.user_features_, ireg
        )

        self.logger.info(
            "[%s] training biased MF model with ALS for %d features", timer, self.features
        )
        start = timer.elapsed()

        with make_progress(self.logger, "BiasedMF", self.epochs, unit="epoch") as epb:
            for epoch in range(self.epochs):
                epoch = epoch + 1

                du = self.als_half_epoch(epoch, u_ctx)
                self.logger.debug("[%s] finished user epoch %d", timer, epoch)

                di = self.als_half_epoch(epoch, i_ctx)
                self.logger.debug("[%s] finished item epoch %d", timer, epoch)

                self.logger.info(
                    "[%s] finished epoch %d (|ΔP|=%.3f, |ΔQ|=%.3f)", timer, epoch, du, di
                )
                epb.update()
                yield self

        if not self.save_user_features:
            self.user_features_ = None
            self.user_ = None

        end = timer.elapsed()
        self.logger.info(
            "[%s] trained %d epochs (%.1fs/epoch)",
            timer,
            self.epochs,
            (end - start) / self.epochs,
        )

    @abstractmethod
    def prepare_data(self, data: Dataset) -> TrainingData:  # pragma: no cover
        """
        Prepare data for training this model.  This takes in the ratings, and is
        supposed to do two things:

        -   Normalize or transform the rating/interaction data, as needed, for
            training.
        -   Store any parameters learned from the normalization (e.g. means) in
            the appropriate member variables.
        -   Return the training data object to use for model training.
        """
        ...

    def initialize_params(self, data: TrainingData):
        """
        Initialize the model parameters at the beginning of training.
        """
        rng = random_generator(self.rng)
        self.logger.debug("initializing item matrix")
        self.item_features_ = self.initial_params(data.n_items, self.features, rng)
        self.logger.debug("|Q|: %f", torch.norm(self.item_features_, "fro"))

        self.logger.debug("initializing user matrix")
        self.user_features_ = self.initial_params(data.n_users, self.features, rng)
        self.logger.debug("|P|: %f", torch.norm(self.user_features_, "fro"))

    @abstractmethod
    def initial_params(
        self, nrows: int, ncols: int, rng: np.random.Generator
    ) -> torch.Tensor:  # pragma: no cover
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
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        user_id = query.user_id
        user_num = None
        if user_id is not None and self.users_ is not None:
            user_num = self.users_.number(user_id, missing=None)

        u_offset = None
        u_feat = None
        if query.user_items is not None and len(query.user_items) > 0:
            u_feat, u_offset = self.new_user_embedding(user_num, query.user_items)

        if u_feat is None:
            if user_num is None or self.user_features_ is None:
                return ItemList(items, scores=np.nan)
            u_feat = self.user_features_[user_num, :]

        item_nums = items.numbers("torch", vocabulary=self.items_, missing="negative")
        item_mask = item_nums >= 0
        i_feats = self.item_features_[item_nums[item_mask], :]

        scores = torch.full((len(items),), np.nan, dtype=torch.float64)
        scores[item_mask] = i_feats @ u_feat

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
