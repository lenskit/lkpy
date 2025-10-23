# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd

from pytest import approx

from lenskit.data import ItemList
from lenskit.metrics.ranking import Hit
from lenskit.testing import demo_recs  # noqa: F401

_log = logging.getLogger(__name__)


def _test_hit(items, rel, **kwargs) -> float:
    recs = ItemList(items, ordered=True)
    truth = ItemList(rel)
    metric = Hit(**kwargs)
    return metric.measure_list(recs, truth)


def test_hit_empty_zero():
    hr = _test_hit([], [1, 3])
    assert hr == 0


def test_hit_norel_na():
    hr = _test_hit([1, 3], [])
    assert np.isnan(hr)


def test_hit_simple_cases():
    hr = _test_hit([1, 3], [1, 3])
    assert hr == 1

    hr = _test_hit([1], [1, 3])
    assert hr == 1

    hr = _test_hit([1, 2, 3, 4], [1, 3])
    assert hr == 1

    hr = _test_hit([1, 2, 3, 4], range(5, 10))
    assert hr == 0

    hr = _test_hit([1, 2, 3, 4], range(4, 9))
    assert hr == 1


def test_hit_series():
    hr = _test_hit(pd.Series([1, 3]), pd.Series([1, 3]))
    assert hr == 1

    hr = _test_hit(pd.Series([1, 2, 3]), pd.Series([1, 3, 5, 7]))
    assert hr == 1

    hr = _test_hit(pd.Series([1, 2, 3]), pd.Series([5, 7]))
    assert hr == 0


def test_hit_series_set():
    hr = _test_hit(pd.Series([1, 2, 3, 4]), [1, 3, 5, 7])
    assert hr == 1

    hr = _test_hit(pd.Series([1, 2, 3]), range(4, 9))
    assert hr == 0


def test_hit_series_index():
    hr = _test_hit(pd.Series([1, 3]), pd.Index([1, 3]))
    assert hr == 1

    hr = _test_hit(pd.Series([1, 2, 3, 4]), pd.Index([1, 3, 5, 7]))
    assert hr == 1

    hr = _test_hit(pd.Series([1, 2, 3]), pd.Index(range(4, 9)))
    assert hr == 0


def test_hit_series_array():
    hr = _test_hit(pd.Series([1, 3]), np.array([1, 3]))
    assert hr == 1

    hr = _test_hit(pd.Series([1, 2, 3, 4]), np.array([1, 3, 5, 7]))
    assert hr == 1

    hr = _test_hit(pd.Series([1, 2, 3]), np.arange(4, 9, 1, "u4"))
    assert hr == 0


def test_hit_array():
    hr = _test_hit(np.array([1, 3]), np.array([1, 3]))
    assert hr == 1

    hr = _test_hit(np.array([1, 2, 3, 4]), np.array([1, 3, 5, 7]))
    assert hr == 1

    hr = _test_hit(np.array([1, 2, 3]), np.arange(4, 9, 1, "u4"))
    assert hr == 0


def test_hit_long_items():
    rel = np.arange(100)
    items = [1, 0, 150, 3, 10, 30, 120, 4, 17]
    items = np.array(items)

    r = _test_hit(items, rel, n=5)
    assert r == 1

    items += 200
    items[5] = 5

    r = _test_hit(np.array(items) + 200, rel, n=5)
    assert r == 0


def test_hit_partial_rel():
    rel = np.arange(100)
    items = [1, 0, 150, 3, 10]

    r = _test_hit(items, rel, n=10)
    assert r == 1
