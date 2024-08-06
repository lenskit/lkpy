# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from typing_extensions import Literal, override

from lenskit.data.dataset import Dataset
from lenskit.data.vocab import Vocabulary

try:
    from sklearn.decomposition import TruncatedSVD

    SKL_AVAILABLE = True
except ImportError:
    SKL_AVAILABLE = False

from ..util import Stopwatch
from . import Predictor
from .bias import Bias

_log = logging.getLogger(__name__)


class BiasedSVD(Predictor):
    """
    Biased matrix factorization for implicit feedback using SciKit-Learn's SVD
    solver (:class:`sklearn.decomposition.TruncatedSVD`).  It operates by first
    computing the bias, then computing the SVD of the bias residuals.

    You'll generally want one of the iterative SVD implementations such as
    :class:`lennskit.algorithms.als.BiasedMF`; this is here primarily as an
    example and for cases where you want to evaluate a pure SVD implementation.
    """

    bias: Bias
    factorization: TruncatedSVD
    users_: Vocabulary
    items_: Vocabulary

    def __init__(
        self,
        features,
        *,
        damping=5,
        bias: bool | Bias = True,
        algorithm: Literal["arpack", "randomized"] = "randomized",
    ):
        if not SKL_AVAILABLE:
            raise ImportError("sklearn.decomposition")
        if bias is True:
            self.bias = Bias(damping=damping)
        elif bias:
            self.bias = bias
        else:
            raise ValueError("no bias configured")
        self.factorization = TruncatedSVD(features, algorithm=algorithm)

    @override
    def fit(self, data: Dataset, **kwargs):
        timer = Stopwatch()
        _log.info("[%s] computing bias", timer)
        self.bias.fit(data)

        g_bias = self.bias.mean_
        u_bias = self.bias.user_offsets_
        i_bias = self.bias.item_offsets_

        _log.info("[%s] sparsifying and normalizing matrix", timer)
        r_mat = data.interaction_matrix("scipy", field="rating", layout="coo", legacy=True)
        # copy the data and start subtracting
        r_mat.data = r_mat.data - g_bias
        if i_bias is not None:
            r_mat.data -= i_bias.values[r_mat.col]
        if u_bias is not None:
            r_mat.data -= u_bias.values[r_mat.row]

        r_mat = r_mat.tocsr()

        _log.info("[%s] training SVD (k=%d)", timer, self.factorization.n_components)  # type: ignore
        Xt = self.factorization.fit_transform(r_mat)
        self.user_components_ = Xt
        self.users_ = data.users.copy()
        self.items_ = data.items.copy()
        _log.info("finished model training in %s", timer)

    def predict_for_user(self, user, items, ratings=None):
        items = np.array(items)
        if user not in self.users_:
            return pd.Series(np.nan, index=items)

        # Get index for user & usable items
        uidx = self.users_.number(user)
        iidx = self.items_.numbers(items, missing="negative")
        good_iidx = iidx[iidx >= 0]

        _log.debug("reverse-transforming user %s (idx=%d)", user, uidx)
        Xt = self.user_components_[[uidx], :]
        X = self.factorization.inverse_transform(Xt)
        # restrict to usable desired items
        Xsel = X[0, good_iidx]
        # convert to output format and de-normalize
        scores = pd.Series(Xsel, index=items[iidx >= 0]).reindex(items)
        scores += self.bias.predict_for_user(user, items, ratings)
        return scores

    def get_params(self, deep=True):
        params = {
            "features": self.factorization.n_components,
            "algorithm": self.factorization.algorithm,
        }
        if deep and self.bias:
            for k, v in self.bias.get_params(True).items():
                params["bias__" + k] = v
        else:
            params["bias"] = self.bias
        return params

    def __str__(self):
        return f"BiasedSVD({self.factorization})"
