# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Literal, override

from lenskit.basic import BiasModel, Damping
from lenskit.data import Dataset, ItemList, QueryInput, RecQuery
from lenskit.data.vocab import Vocabulary
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions
from lenskit.util import Stopwatch

try:
    from sklearn.decomposition import TruncatedSVD

    SKL_AVAILABLE = True
except ImportError:
    SKL_AVAILABLE = False


_log = logging.getLogger(__name__)


@dataclass
class BiasedSVDConfig:
    features: int = 50
    damping: Damping = 5
    algorithm: Literal["arpack", "randomized"] = "randomized"
    n_iter: int = 5


class BiasedSVDScorer(Component[ItemList], Trainable):
    """
    Biased matrix factorization for explicit feedback using SciKit-Learn's
    :class:`~sklearn.decomposition.TruncatedSVD`.  It operates by first
    computing the bias, then computing the SVD of the bias residuals.

    You'll generally want one of the iterative SVD implementations such as
    :class:`lennskit.algorithms.als.BiasedMFScorer`; this is here primarily as
    an example and for cases where you want to evaluate a pure SVD
    implementation.

    Stability:
        Caller
    """

    config: BiasedSVDConfig

    bias_: BiasModel
    factorization_: TruncatedSVD
    users_: Vocabulary
    items_: Vocabulary
    user_components_: NDArray[np.float64]

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        if hasattr(self, "factorization_") and not options.retrain:
            return

        timer = Stopwatch()
        _log.info("[%s] computing bias", timer)
        self.bias_ = BiasModel.learn(data, self.config.damping)

        g_bias = self.bias_.global_bias
        u_bias = self.bias_.user_biases
        i_bias = self.bias_.item_biases

        _log.info("[%s] sparsifying and normalizing matrix", timer)
        r_mat = data.interaction_matrix(format="scipy", field="rating", layout="coo", legacy=True)
        # copy the data and start subtracting
        r_mat.data = r_mat.data - g_bias
        if i_bias is not None:
            r_mat.data -= i_bias[r_mat.col]
        if u_bias is not None:
            r_mat.data -= u_bias[r_mat.row]

        r_mat = r_mat.tocsr()

        self.factorization_ = TruncatedSVD(
            self.config.features, algorithm=self.config.algorithm, n_iter=self.config.n_iter
        )
        _log.info("[%s] training SVD (k=%d)", timer, self.factorization_.n_components)  # type: ignore
        Xt = self.factorization_.fit_transform(r_mat)  # type: ignore
        self.user_components_ = Xt
        self.users_ = data.users
        self.items_ = data.items
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
