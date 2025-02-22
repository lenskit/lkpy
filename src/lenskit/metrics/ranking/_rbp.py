# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
from typing_extensions import override

from lenskit.data.items import ItemList

from ._base import ListMetric, RankingMetricBase


class RBP(ListMetric, RankingMetricBase):
    """
    Evaluate recommendations with rank-biased precision :cite:p:`rbp` with a
    patience parameter :math:`\\gamma`.

    If :math:`r_{ui} \\in \\{0, 1\\}` is binary implicit ratings, this is
    computed by:

    .. math::
        \\begin{align*}
        \\operatorname{RBP}_\\gamma(L, u) & =(1 - \\gamma) \\sum_i r_{ui} p^i
        \\end{align*}

    The original RBP metric depends on the idea that the rank-biased sum of
    binary relevance scores in an infinitely-long, perfectly-precise list has is
    :math:`1/(1 - \\gamma)`. However, in recommender evaluation, we usually have
    a small test set, so the maximum achievable RBP is significantly less, and
    is a function of the number of test items.  With ``normalize=True``, the RBP
    metric will be normalized by the maximum achievable with the provided test
    data.

    :cite:t:`rbp` provide an extended discussion on choosing the patience
    parameter :math:`\\gamma`.  This metric defaults to :math:`\\gamma=0.85`, to
    provide a relatively shallow curve and reward good items on the first few
    pages of results (in a 10-per-page setting).  Recommendation systems data
    has no pooling, so the variance of this estimator may be high as they note
    in the paper; however, RBP with high patience should be no worse than nDCG
    (and perhaps even better) in this regard.

    .. warning::

        The additional normalization is experimental, and should not yet be used
        for published research results.

    Args:
        k:
            The maximum recommendation list length.
        patience:
            The patience parameter :math:`\\gamma`, the probability that the
            user continues browsing at each point.  The default is 0.85.
        normalize:
            Whether to normalize the RBP scores; if ``True``, divides the RBP
            score by the maximum achievable with the test data (as in nDCG).

    Stability:
        Caller
    """

    patience: float
    normalize: bool

    def __init__(self, k: int | None = None, *, patience: float = 0.85, normalize: bool = False):
        super().__init__(k)
        self.patience = patience
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

        items = recs.ids()
        good = np.isin(items, test.ids())
        # γ^(r-1)
        disc = np.power(self.patience, np.arange(k))
        rbp = np.sum(disc[good])
        if self.normalize:
            # normalize by max achieveable RBP
            max = np.sum(disc[: min(nrel, k)])
            return rbp / max
        else:
            # standard RBP normalization
            return rbp * (1 - self.patience)
