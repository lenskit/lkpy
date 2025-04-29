# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
from typing_extensions import override

from lenskit.data.items import ItemList

from ._base import ListMetric, RankingMetricBase


class AveragePrecision(ListMetric, RankingMetricBase):
    """
    Compute Average Precision (AP) for a single user's recommendations.  This is
    the average of the precision at each relevant item in the ranked list.
    """

    @property
    def label(self):
        if self.k is not None:
            return f"AveragePrecision@{self.k}"
        else:
            return "AveragePrecision"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)
        nrecs = len(recs)
        if nrecs == 0:
            return np.nan

        items = recs.ids()
        ap_sum = 0.0
        good = np.isin(items, test.ids())
        sum_good = np.cumsum(good)            # cumulative count of “good” up to each rank
        ranks = np.arange(1, len(items) + 1)  # ranks starting from 1
        ap_sum = np.sum(sum_good[good] / ranks[good])
        
        return ap_sum / min(len(test.ids()), items)