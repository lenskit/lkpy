# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import inspect
import logging

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.recommender_base import RecommenderBase
from scipy.sparse import csr_matrix
from typing_extensions import override

from lenskit.algorithms import Predictor, Recommender
from lenskit.data.dataset import Dataset
from lenskit.data.vocab import Vocabulary

_logger = logging.getLogger(__name__)

__all__ = [
    "BaseRec",
    "ALS",
    "BPR",
]


class BaseRec(Recommender, Predictor):
    """
    Base class for Implicit-backed recommenders.

    Args:
        delegate(implicit.RecommenderBase):
            The delegate algorithm.

    Attributes:
        delegate(implicit.RecommenderBase):
            The :py:mod:`implicit` delegate algorithm.
        matrix_(scipy.sparse.csr_matrix):
            The user-item rating matrix.
        user_index_(pandas.Index):
            The user index.
        item_index_(pandas.Index):
            The item index.
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

    @override
    def fit(self, data: Dataset, **kwargs):
        matrix = data.interaction_matrix("scipy", layout="csr", legacy=True)
        uir = matrix * self.weight
        if getattr(self.delegate, "item_factors", None) is not None:  # pragma: no cover
            _logger.warn("implicit algorithm already trained, re-fit is usually a bug")

        _logger.info("training %s on %s matrix (%d nnz)", self.delegate, uir.shape, uir.nnz)

        self.delegate.fit(uir)

        self.matrix_ = matrix
        self.users_ = data.users
        self.items_ = data.items

        return self

    @override
    def recommend(self, user, n: int | None = None, candidates=None, ratings=None):
        uid = self.users_.number(user, missing=None)
        if uid is None:
            _logger.debug("unknown user %s, cannot recommend", user)
            return pd.DataFrame({"item": []})

        matrix = self.matrix_[[uid], :]

        if candidates is None:
            i_n = n if n is not None else self.items_.size
            _logger.debug("recommending for user %s with unlimited candidates", user)
            recs, scores = self.delegate.recommend(uid, matrix, N=i_n)  # type: ignore
        else:
            cands = self.items_.numbers(candidates, missing="negative")
            cands = cands[cands >= 0]
            _logger.debug("recommending for user %s with %d candidates", user, len(cands))
            recs, scores = self.delegate.recommend(uid, matrix, items=cands)  # type: ignore

        if n is not None:
            recs = recs[:n]
            scores = scores[:n]

        rec_df = pd.DataFrame(
            {
                "item": self.items_.ids(recs),
                "score": scores,
            }
        )
        return rec_df

    @override
    def predict_for_user(self, user, items, ratings=None):
        uid = self.users_.number(user, missing=None)
        if uid is None:
            return pd.Series(np.nan, index=items)

        iids = self.items_.numbers(items, missing="negative")
        iids = iids[iids >= 0]

        ifs = self.delegate.item_factors[iids]
        uf = self.delegate.user_factors[uid]
        # convert back if these are on CUDA
        if hasattr(ifs, "to_numpy"):
            ifs = ifs.to_numpy()
            uf = uf.to_numpy()
        scores = np.dot(ifs, uf.T)
        scores = pd.Series(np.ravel(scores), index=self.items_.ids(iids))
        return scores.reindex(items)

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
