# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

from pytest import approx

from lenskit.data import ItemList
from lenskit.metrics import call_metric
from lenskit.metrics.ranking import GradedRBP


def test_grbp_empty():
    recs = ItemList([], ordered=True)
    truth = ItemList(item_ids=[1, 2, 3], grade=[1.0, 1.0, 1.0])

    metric = GradedRBP()
    assert metric.measure_list(recs, truth) == approx(0.0)
