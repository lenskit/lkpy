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
        if self.n is not None:
            return f"AveragePrecision@{self.n}"
        else:
            return "AveragePrecision"

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)
        nrecs = len(recs)
        if nrecs == 0:
            return np.nan

        ap_sum = 0.0
        good = recs.isin(test)
        sum_good = np.cumsum(good)  # cumulative count of “good” up to each rank
        ranks = recs.ranks()
        assert ranks is not None
        ap_sum = np.sum(sum_good[good] / ranks[good]).item()

        denom = min(len(test), len(recs))
        return ap_sum / denom
