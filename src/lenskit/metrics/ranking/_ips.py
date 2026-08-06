# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
IPS-weighted ranking metrics for missing-not-at-random implicit feedback.
"""

from __future__ import annotations

from typing import override

import numpy as np
import pandas as pd

from lenskit.data import ItemList

from ._base import ListMetric, RankingMetricBase
from ._propensity import PropensityModel
from ._weighting import GeometricRankWeight, RankWeight


class IPSRBP(ListMetric, RankingMetricBase):
    r"""
    Rank-biased precision :cite:p:`rbp` corrected for missing-not-at-random
    implicit feedback by inverse propensity scoring
    :cite:p:`yangUnbiasedOfflineRecommender2018`.

    Logged implicit feedback is biased towards popular items, which are more
    likely to be presented and therefore more likely to be interacted with.
    Measuring over such a log rewards accuracy on popular items more than
    accuracy on long-tail ones.  IPS corrects for this by weighting each
    observed relevant item by the inverse :math:`w_i = 1 / P_i` of its
    propensity to be observed.

    For observed relevant items :math:`S^*_u`, and RBP contribution
    :math:`c(r) = (1 - p) p^{r-1}` at rank :math:`r` (zero past the cutoff),
    this computes the self-normalized (SNIPS) estimator:

    .. math::
        \operatorname{RBP}^{\mathrm{SNIPS}}(L, u) =
        \frac{\sum_{i \in S^*_u} w_i\, c(r_{ui})}{\sum_{i \in S^*_u} w_i}

    The denominator is a control variate with expectation :math:`|S_u|`, the
    size of the user's complete relevant set, which the plain IPS estimator
    requires but cannot observe.  It also makes the score invariant to the
    scale of the propensities.

    With ``self_normalized=False`` the denominator is dropped, giving
    :math:`\sum_i w_i c(r_{ui})`.  This is proportional to the IPS estimator
    only when all users have the same number of relevant items, and its
    variance is unbounded in the weights, but it is the form that reduces
    exactly to :class:`RBP` under uniform propensities.

    The denominator is not truncated by ``n``: :math:`S^*_u` is the whole
    observed relevant set, and only the hits are cut off.  With
    :class:`UniformPropensity`, this metric reduces to the average-over-all
    evaluator, which for RBP is the standard score divided by the number of
    test items.

    Args:
        n:
            The maximum recommendation list length.
        k:
            Deprecated alias for ``n``.
        propensity:
            The observation propensity model.  Estimate propensities from
            training data, not from the test observations being weighted.
        patience:
            The patience parameter :math:`p`, the probability that the user
            continues browsing at each point.
        weight:
            The rank weighting model.  Defaults to
            :class:`GeometricRankWeight` with the given patience, which gives
            RBP; other weightings give the corrected form of the corresponding
            metric.
        self_normalized:
            Whether to use the self-normalized estimator.

    Stability:
        Caller
    """

    propensity: PropensityModel
    patience: float
    weight: RankWeight
    self_normalized: bool

    def __init__(
        self,
        n: int | None = None,
        *,
        k: int | None = None,
        propensity: PropensityModel,
        patience: float = 0.85,
        weight: RankWeight | None = None,
        self_normalized: bool = True,
    ):
        super().__init__(n, k=k)
        self.propensity = propensity
        self.patience = patience
        self.weight = weight if weight is not None else GeometricRankWeight(patience)
        self.self_normalized = self_normalized

    @property
    def label(self):
        if self.n is not None:
            return f"IPSRBP@{self.n}"
        else:
            return "IPSRBP"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        if len(test) == 0:
            return np.nan

        item_weights = self.propensity.weights(test)

        recs = self.truncate(recs)
        ranks = recs.ranks()
        if ranks is None:
            raise TypeError("item list is not ordered")

        rank_weights = self.weight.weight(ranks)

        # the contribution c(r) is the rank weight over its series sum; with no
        # defined series sum, normalize by the list as RBP does
        wmax = self.weight.series_sum()
        normalization = wmax if wmax is not None else np.sum(rank_weights).item()
        if normalization == 0:
            return np.nan

        good = recs.isin(test)
        weights = pd.Series(item_weights, index=test.ids())
        hits = weights.reindex(recs.ids()[good]).to_numpy(dtype=np.float64)
        gain = np.dot(hits, rank_weights[good]).item() / normalization

        if not self.self_normalized:
            return gain

        total = np.sum(item_weights).item()
        return gain / total if total else np.nan
