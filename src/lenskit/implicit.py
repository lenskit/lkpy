# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Bridges to recommendation models from :mod:`implicit`.
"""

import logging

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.recommender_base import RecommenderBase
from numpy.typing import NDArray
from pydantic import BaseModel, JsonValue
from scipy.sparse import csr_matrix
from typing_extensions import override

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_logger = logging.getLogger(__name__)

__all__ = [
    "BaseRec",
    "ALS",
    "BPR",
]


class ImplicitConfig(BaseModel, extra="allow"):
    __pydantic_extra__: dict[str, JsonValue]


class ImplicitALSConfig(ImplicitConfig, extra="allow"):
    weight: float = 40.0


class BaseRec(Component[ItemList], Trainable):
    """
    Base class for Implicit-backed recommenders.

    Stability:
        Caller
    """

    config: ImplicitConfig
    weight: float = 1.0

    matrix: csr_matrix
    """
    The user-item rating matrix from training.
    """
    users: Vocabulary
    """
    The user ID mapping from training.
    """
    item: Vocabulary
    """
    The item ID mapping from training.
    """
    user_embeddings: NDArray[np.float32]
    item_embeddings: NDArray[np.float32]

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        if hasattr(self, "item_embeddings") and not options.retrain:
            return

        matrix = data.interaction_matrix(format="scipy", layout="csr", legacy=True)
        uir = matrix * self.weight

        delegate = self._construct()
        _logger.info("training %s on %s matrix (%d nnz)", delegate, uir.shape, uir.nnz)

        delegate.fit(uir)

        self.matrix = matrix
        self.users = data.users
        self.item = data.items

        if hasattr(delegate.item_factors, "to_numpy"):
            self.user_embeddings = delegate.user_factors.to_numpy()
            self.item_embeddings = delegate.item_factors.to_numpy()
        else:
            self.user_embeddings = delegate.user_factors
            self.item_embeddings = delegate.item_factors

        return self

    def _construct(self) -> RecommenderBase:
        raise NotImplementedError("implicit constructor not implemented")

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        user_num = None
        if query.user_id is not None:
            user_num = self.users.number(query.user_id, missing=None)

        if user_num is None or not len(items):
            return ItemList(items, scores=np.nan)

        inos = items.numbers(vocabulary=self.item, missing="negative")
        mask = inos >= 0
        good_inos = inos[mask]

        ifs = self.item_embeddings[good_inos]
        uf = self.user_embeddings[user_num]

        scores = np.full(len(items), np.nan)
        scores[mask] = np.dot(ifs, uf.T).reshape(-1)

        return ItemList(items, scores=scores)


class ALS(BaseRec):
    """
    LensKit interface to :py:mod:`implicit.cpu.als` (or GPU version).

    Stability:
        Caller
    """

    config: ImplicitALSConfig

    @property
    def weight(self):
        return self.config.weight

    def _construct(self):
        return AlternatingLeastSquares(**self.config.__pydantic_extra__)  # type: ignore


class BPR(BaseRec):
    """
    LensKit interface to :py:mod:`implicit.cpu.bpr` (or GPU version).

    Stability:
        Caller
    """

    def _construct(self):
        return BayesianPersonalizedRanking(**self.config.__pydantic_extra__)  # type: ignore
