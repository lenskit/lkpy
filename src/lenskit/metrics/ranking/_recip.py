# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
from typing_extensions import override

from lenskit.data import ItemList

from ._base import ListMetric, RankingMetricBase


class RecipRank(ListMetric, RankingMetricBase):
    """
    Compute the reciprocal rank :cite:p:`trec5-confusion` of the first relevant
    item in a list of recommendations.  Taking the mean of this metric over the
    recommendation lists in a run yields the MRR (mean reciprocal rank).

    Let :math:`\\kappa` denote the 1-based rank of the first relevant item in
    :math:`L`, with :math:`\\kappa=\\infty` if none of the first :math:`k` items
    in :math:`L` are relevant; then the reciprocal rank is :math:`1 / \\kappa`.
    If no elements are relevant, the reciprocal rank is therefore 0.
    :cite:t:`deshpande:iknn` call this the “reciprocal hit rate”.

    Stability:
        Caller
    """

    @property
    def label(self):
        if self.k is not None:
            return f"RecipRank@{self.k}"
        else:
            return "RecipRank"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        if len(test) == 0:
            return np.nan

        recs = self.truncate(recs)
        items = recs.ids()
        good = np.isin(items, test.ids())
        (npz,) = np.nonzero(good)
        if len(npz):
            return 1.0 / (npz[0] + 1.0)
        else:
            return 0.0
