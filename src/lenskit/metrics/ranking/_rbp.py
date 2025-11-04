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


def rank_biased_precision(
    good: np.ndarray, weights: np.ndarray, normalization: float = 1.0
) -> float:
    """
    Compute rank-biased precision given explicit weights.

    Args:
        good:
            Boolean array indicating relevant items at each position.
        weights:
            Weight for each item position (same length as good).
        normalization:
            Optional normalization factor, defaults to 1.0.

    Returns:
        RBP score
    """

    rbp = np.sum(weights[good]).item()

    return rbp / normalization


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
        n:
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
        weight_field:
            Name of a field in the item list to use as weights. If provided,
            weights are read from this field instead of being computed from
            the rank model.

    Stability:
        Caller
    """

    weight: RankWeight | None
    patience: float
    normalize: bool
    weight_field: str | None

    def __init__(
        self,
        n: int | None = None,
        *,
        k: int | None = None,
        weight: RankWeight | None = None,
        patience: float = 0.85,
        normalize: bool = False,
        weight_field: str | None = None,
    ):
        super().__init__(n, k=k)
        self.patience = patience
        if weight is None and weight_field is None:
            weight = GeometricRankWeight(patience)
        self.weight = weight
        self.normalize = normalize
        self.weight_field = weight_field

    @property
    def label(self):
        if self.n is not None:
            return f"RBP@{self.n}"
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

        if self.weight_field is not None:
            # use custom weights from field
            weights = recs.field(self.weight_field)
            normalization = np.sum(weights).item()

        else:
            ranks = recs.ranks()
            assert ranks is not None

            weights = self.weight.weight(ranks)

            # figure out normalization
            wmax = self.weight.series_sum()
            if self.normalize:
                # normalize by max achieveable RBP
                normalization = np.sum(weights[: min(nrel, k)]).item()
            elif wmax is not None:
                # normalization defined by metric
                normalization = wmax
            else:
                normalization = np.sum(weights).item()

        return rank_biased_precision(good, weights, normalization)
