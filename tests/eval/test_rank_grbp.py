# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

from pytest import approx

from lenskit.data import ItemList
from lenskit.metrics.ranking import RBP, GradedRBP


def test_grbp_empty():
    recs = ItemList([], ordered=True)
    truth = ItemList(item_ids=[1, 2, 3], grade=[1.0, 1.0, 1.0])

    metric = GradedRBP()
    assert metric.measure_list(recs, truth) == approx(0.0)


def test_grbp_unknown_grade():
    recs = ItemList([1, 2], ordered=True)
    truth = ItemList(item_ids=[1], grade=[1.0])

    p = 0.5
    metric = GradedRBP(patience=p, unknown_grade=0.25)

    expected = (1 - p) * (1 + 0.25 * p)

    assert metric.measure_list(recs, truth) == approx(expected)


def test_grbp_scaling():
    recs = ItemList([1, 2], ordered=True)
    truth = ItemList(item_ids=[1, 2], grade=[2.0, 4.0])

    p = 0.5
    metric = GradedRBP(patience=p, scale=True)

    scaled = np.array([0.5, 1.0])
    expected = (1 - p) * (scaled[0] + scaled[1] * p)

    assert metric.measure_list(recs, truth) == approx(expected)


def test_grbp_binary():
    recs = ItemList([1, 2, 3], ordered=True)

    graded_truth = ItemList(item_ids=[1, 3], graded=[1.0, 1.0])
    binary_truth = ItemList([1, 3])  # no grade field

    grbp = GradedRBP()
    rbp = RBP()

    assert grbp.measure_list(recs, graded_truth) == approx(rbp.measure_list(recs, binary_truth))
