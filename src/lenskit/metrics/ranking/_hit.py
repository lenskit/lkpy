# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
from typing_extensions import override

from lenskit.data import ItemList

from ._base import ListMetric, RankingMetricBase


class Hit(ListMetric, RankingMetricBase):
    """
    Compute whether or not a list is a hit; any list with at least one
    relevant item in the first :math:`k` positions (:math:`L_{\\le k} \\cap
    I_u^{\\mathrm{test}} \\ne \\emptyset`) is scored as 1, and lists with no
    relevant items as 0.  When averaged over the recommendation lists, this
    computes the *hit rate* :cite:p:`deshpande:iknn`.

    Stability:
        Caller
    """

    @property
    def label(self):
        if self.n is not None:
            return f"Hit@{self.n}"
        else:
            return "Hit"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        if len(test) == 0:
            return np.nan

        recs = self.truncate(recs)

        return 1 if np.any(recs.isin(test)) else 0
