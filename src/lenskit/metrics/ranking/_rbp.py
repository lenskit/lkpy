# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
from typing_extensions import override

from lenskit.data.items import ItemList

from ._base import ListMetric, RankingMetricBase
from ._weighting import GeometricRankWeight, RankWeight


class RBP(ListMetric, RankingMetricBase):
    """
    Evaluate recommendations with rank-biased precision :cite:p:`rbp`.

    If :math:`r_{ui} \\in \\{0, 1\\}` is binary implicit ratings, and the
    weighting is the default geometric weight with patience :math:`p`, the RBP
    is computed by:

    .. math::
        \\begin{align*}
        \\operatorname{RBP}_p(L, u) & =(1 - p) \\sum_i r_{ui} p^i
        \\end{align*}

    The original RBP metric depends on the idea that the rank-biased sum of
    binary relevance scores in an infinitely-long, perfectly-precise list has is
    :math:`1/(1 - p)`.  If RBP is used with a non-standard weighting that does
    not have a defined infinite series sum, then this metric will normalize by
    the sum of the discounts for the recommendation list.

    :cite:t:`rbp` provide an extended discussion on choosing the patience
    parameter :math:`\\gamma`.  This metric defaults to :math:`\\gamma=0.85`, to
    provide a relatively shallow curve and reward good items on the first few
    pages of results (in a 10-per-page setting).  Recommendation systems data
    has no pooling, so the variance of this estimator may be high as they note
    in the paper; however, RBP with high patience should be no worse than nDCG
    (and perhaps even better) in this regard.

    In recommender evaluation, we usually have a small test set, so the maximum
    achievable RBP is significantly less than the theoretical maximum, and is a
    function of the number of test items.  With ``normalize=True``, the RBP
    metric will be normalized by the maximum achievable with the provided test
    data, like NDCG.

    .. warning::

        The additional normalization is experimental, and should not yet be used
        for published research results.

    Args:
        k:
            The maximum recommendation list length.
        weight:
            The rank weighting model to use.  Defaults to
            :class:`GeometricRankWeight` with the specified patience parameter.
        patience:
            The patience parameter :math:`p`, the probability that the user
            continues browsing at each point.  The default is 0.85.
        normalize:
            Whether to normalize the RBP scores; if ``True``, divides the RBP
            score by the maximum achievable with the test data (as in nDCG).

    Stability:
        Caller
    """

    weight: RankWeight
    patience: float
    normalize: bool

    def __init__(
        self,
        k: int | None = None,
        *,
        weight: RankWeight | None = None,
        patience: float = 0.85,
        normalize: bool = False,
    ):
        super().__init__(k)
        self.patience = patience
        if weight is None:
            weight = GeometricRankWeight(patience)
        self.weight = weight
        self.normalize = normalize

    @property
    def label(self):
        if self.k is not None:
            return f"RBP@{self.k}"
        else:
            return "RBP"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)
        k = len(recs)

        nrel = len(test)
        if nrel == 0:
            return np.nan

        good = recs.isin(test)
        weight = self.weight.weight(np.arange(1, k + 1))
        rbp = np.sum(weight[good]).item()
        wmax = self.weight.series_sum()
        if self.normalize:
            # normalize by max achieveable RBP
            max = np.sum(weight[: min(nrel, k)]).item()
            return rbp / max
        elif wmax is not None:
            # normalization defined by metric
            return rbp / wmax
        else:
            # normalize by total weight
            return rbp / np.sum(weight).item()
