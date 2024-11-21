# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import inspect
import logging

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.recommender_base import RecommenderBase
from scipy.sparse import csr_matrix
from typing_extensions import override

from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.pipeline import Component, Trainable

_logger = logging.getLogger(__name__)

__all__ = [
    "BaseRec",
    "ALS",
    "BPR",
]


class BaseRec(Component, Trainable):
    """
    Base class for Implicit-backed recommenders.

    Args:
        delegate:
            The delegate algorithm.
    """

    delegate: RecommenderBase
    """
    The delegate algorithm from :mod:`implicit`.
    """
    weight: float
    """
    The weight for positive examples (only used by some algorithms).
    """

    matrix_: csr_matrix
    """
    The user-item rating matrix from training.
    """
    users_: Vocabulary
    """
    The user ID mapping from training.
    """
    items_: Vocabulary
    """
    The item ID mapping from training.
    """

    def __init__(self, delegate: RecommenderBase):
        self.delegate = delegate
        self.weight = 1.0

    @property
    def is_trained(self):
        return hasattr(self, "matrix_")

    @override
    def train(self, data: Dataset):
        matrix = data.interaction_matrix("scipy", layout="csr", legacy=True)
        uir = matrix * self.weight
        if getattr(self.delegate, "item_factors", None) is not None:  # pragma: no cover
            _logger.warning("implicit algorithm already trained, re-fit is usually a bug")

        _logger.info("training %s on %s matrix (%d nnz)", self.delegate, uir.shape, uir.nnz)

        self.delegate.fit(uir)

        self.matrix_ = matrix
        self.users_ = data.users
        self.items_ = data.items

        return self

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        user_num = None
        if query.user_id is not None:
            user_num = self.users_.number(query.user_id, missing=None)

        if user_num is None:
            return ItemList(items, scores=np.nan)

        inos = items.numbers(vocabulary=self.items_, missing="negative")
        mask = inos >= 0
        good_inos = inos[mask]

        ifs = self.delegate.item_factors[good_inos]
        uf = self.delegate.user_factors[user_num]

        # convert back if these are on CUDA
        if hasattr(ifs, "to_numpy"):
            ifs = ifs.to_numpy()
            uf = uf.to_numpy()

        scores = np.full(len(items), np.nan)
        scores[mask] = np.dot(ifs, uf.T)

        return ItemList(items, scores=scores)

    def __getattr__(self, name):
        if "delegate" not in self.__dict__:
            raise AttributeError()
        dd = self.delegate.__dict__
        if name in dd:
            return dd[name]
        else:
            raise AttributeError()

    def get_params(self, deep=True):
        dd = self.delegate.__dict__
        sig = inspect.signature(self.delegate.__class__)
        names = list(sig.parameters.keys())
        return dict([(k, dd.get(k)) for k in names])

    def __str__(self):
        return "Implicit({})".format(self.delegate)


class ALS(BaseRec):
    """
    LensKit interface to :py:mod:`implicit.cpu.als` (or GPU version).
    """

    def __init__(self, *args, weight=40.0, **kwargs):
        """
        Construct an ALS recommender.  The arguments are passed as-is to
        :py:class:`implicit.als.AlternatingLeastSquares`.  The `weight`
        parameter controls the confidence weight for positive examples.
        """

        super().__init__(AlternatingLeastSquares(*args, **kwargs))
        self.weight = weight


class BPR(BaseRec):
    """
    LensKit interface to :py:mod:`implicit.cpu.bpr` (or GPU version).
    """

    def __init__(self, *args, **kwargs):
        """
        Construct a BPR recommender.  The arguments are passed as-is to
        :py:class:`implicit.als.BayesianPersonalizedRanking`.
        """
        super().__init__(BayesianPersonalizedRanking(*args, **kwargs))
