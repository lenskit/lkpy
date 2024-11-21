# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Literal, override

from lenskit.basic import BiasModel
from lenskit.data import Dataset, ItemList, QueryInput, RecQuery, UITuple
from lenskit.data.vocab import Vocabulary
from lenskit.pipeline import Component, Trainable
from lenskit.util import Stopwatch

try:
    from sklearn.decomposition import TruncatedSVD

    SKL_AVAILABLE = True
except ImportError:
    SKL_AVAILABLE = False


_log = logging.getLogger(__name__)


class BiasedSVD(Component, Trainable):
    """
    Biased matrix factorization for implicit feedback using SciKit-Learn's SVD
    solver (:class:`sklearn.decomposition.TruncatedSVD`).  It operates by first
    computing the bias, then computing the SVD of the bias residuals.

    You'll generally want one of the iterative SVD implementations such as
    :class:`lennskit.algorithms.als.BiasedMF`; this is here primarily as an
    example and for cases where you want to evaluate a pure SVD implementation.
    """

    features: int
    damping: UITuple[float]
    algorithm: Literal["arpack", "randomized"]
    n_iter: int

    bias_: BiasModel
    factorization_: TruncatedSVD
    users_: Vocabulary
    items_: Vocabulary
    user_components_: NDArray[np.float64]

    def __init__(
        self,
        features: int,
        *,
        damping: UITuple[float] | float | tuple[float, float] = 5,
        algorithm: Literal["arpack", "randomized"] = "randomized",
        n_iter: int = 5,
    ):
        self.features = features
        self.damping = UITuple.create(damping)
        self.algorithm = algorithm
        self.n_iter = n_iter

    @property
    def is_trained(self):
        return hasattr(self, "factorization_")

    @override
    def train(self, data: Dataset):
        timer = Stopwatch()
        _log.info("[%s] computing bias", timer)
        self.bias_ = BiasModel.learn(data, self.damping)

        g_bias = self.bias_.global_bias
        u_bias = self.bias_.user_biases
        i_bias = self.bias_.item_biases

        _log.info("[%s] sparsifying and normalizing matrix", timer)
        r_mat = data.interaction_matrix("scipy", field="rating", layout="coo", legacy=True)
        # copy the data and start subtracting
        r_mat.data = r_mat.data - g_bias
        if i_bias is not None:
            r_mat.data -= i_bias[r_mat.col]
        if u_bias is not None:
            r_mat.data -= u_bias[r_mat.row]

        r_mat = r_mat.tocsr()

        self.factorization_ = TruncatedSVD(
            self.features, algorithm=self.algorithm, n_iter=self.n_iter
        )
        _log.info("[%s] training SVD (k=%d)", timer, self.factorization_.n_components)  # type: ignore
        Xt = self.factorization_.fit_transform(r_mat)
        self.user_components_ = Xt
        self.users_ = data.users.copy()
        self.items_ = data.items.copy()
        _log.info("finished model training in %s", timer)

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)

        uidx = None
        if query.user_id is not None:
            uidx = self.users_.number(query.user_id, missing="none")

        if uidx is None:
            return ItemList(items, scores=np.nan)

        # Get index for user & usable items
        iidx = items.numbers(vocabulary=self.items_, missing="negative")
        good_iidx = iidx[iidx >= 0]

        _log.debug("reverse-transforming user %s (idx=%d)", query.user_id, uidx)
        Xt = self.user_components_[[uidx], :]
        X = self.factorization_.inverse_transform(Xt)
        # restrict to usable desired items
        Xsel = X[0, good_iidx]

        scores = np.full(len(items), np.nan)
        scores[iidx >= 0] = Xsel

        biases, _ub = self.bias_.compute_for_items(items, query.user_id, query.user_items)
        scores += biases

        return ItemList(items, scores=scores)

    def __str__(self):
        return f"BiasedSVD({self.features})"
