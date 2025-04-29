# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

import pytest

from lenskit.data import ItemList
from lenskit.metrics.ranking import AveragePrecision


def _test_ap(items, rel, **k):
    recs = ItemList(items, ordered=True)
    truth = ItemList(rel)
    return AveragePrecision(**k).measure_list(recs, truth)


def test_ap_empty_none():
    ap = _test_ap([], [1, 2, 3])
    assert np.isnan(ap)

    ap = _test_ap([], [10, 20, 30])
    assert np.isnan(ap)


def test_ap_all_relevant():
    ap = _test_ap([1, 3, 5], [1, 3, 5])
    assert ap == pytest.approx(1.0)

    ap = _test_ap([1, 5, 7], [1, 5, 7])
    assert ap == pytest.approx(1.0)


def test_ap_partial_hits():
    ap = _test_ap([1, 2, 3], [1, 3])
    assert ap == pytest.approx((1.0 + 2 / 3) / 2)

    ap = _test_ap([1, 2, 3], [2, 4])
    assert ap == pytest.approx(0.25)


def test_ap_cutoff_k():
    ap = _test_ap([1, 2, 3, 4, 5], [1, 3, 5], k=3)
    assert ap == pytest.approx((1.0 + 2 / 3) / 3)

    ap = _test_ap([1, 2, 3, 4, 5], [4, 5, 6], k=2)
    assert ap == pytest.approx(0.0)


def test_ap_no_hits():
    ap = _test_ap([10, 20, 30], [1, 2, 3])
    assert ap == pytest.approx(0.0)

    ap = _test_ap([1, 4, 6], [2, 3, 5])
    assert ap == pytest.approx(0.0)
