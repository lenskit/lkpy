# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
from typing_extensions import override

from lenskit.data.items import ItemList

from ._base import ListMetric, RankingMetricBase


class Precision(ListMetric, RankingMetricBase):
    """
    Compute recommendation precision.  This is computed as:

    .. math::
        \\frac{|L \\cap I_u^{\\mathrm{test}}|}{|L|}

    In the uncommon case that ``k`` is specified and ``len(recs) < k``, this metric uses
    ``len(recs)`` as the denominator.

    Stability:
        Caller
    """

    @property
    def label(self):
        if self.n is not None:
            return f"Precision@{self.n}"
        else:
            return "Precision"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)
        nrecs = len(recs)
        if nrecs == 0:
            return np.nan

        ngood = recs.isin(test).sum()
        return ngood / nrecs


class Recall(ListMetric, RankingMetricBase):
    """
    Compute recommendation recall.  This is computed as:

    .. math::
        \\frac{|L \\cap I_u^{\\mathrm{test}}|}{\\operatorname{min}\\{|I_u^{\\mathrm{test}}|, k\\}}
    """

    @property
    def label(self):
        if self.n is not None:
            return f"Recall@{self.n}"
        else:
            return "Recall"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)

        ngood = recs.isin(test).sum()
        nrel = len(test)
        if self.n is not None and self.n < nrel:
            nrel = self.n
        return ngood / nrel
