# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from pytest import approx

from lenskit.data import ItemList
from lenskit.metrics.ranking import Precision
from lenskit.testing import demo_recs  # noqa: F401


def _test_prec(items, rel, **k):
    recs = ItemList(items, ordered=True)
    truth = ItemList(rel)
    return Precision(**k).measure_list(recs, truth)


def test_precision_empty_none():
    prec = _test_prec([], [1, 3])
    assert np.isnan(prec)


def test_precision_simple_cases():
    prec = _test_prec([1, 3], [1, 3])
    assert prec == approx(1.0)

    prec = _test_prec([1], [1, 3])
    assert prec == approx(1.0)

    prec = _test_prec([1, 2, 3, 4], [1, 3])
    assert prec == approx(0.5)

    prec = _test_prec([1, 2, 3, 4], [1, 3, 5])
    assert prec == approx(0.5)

    prec = _test_prec([1, 2, 3, 4], range(5, 10))
    assert prec == approx(0.0)

    prec = _test_prec([1, 2, 3, 4], range(4, 10))
    assert prec == approx(0.25)


def test_precision_series():
    prec = _test_prec(pd.Series([1, 3]), pd.Series([1, 3]))
    assert prec == approx(1.0)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), pd.Series([1, 3, 5]))
    assert prec == approx(0.5)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), pd.Series(range(4, 10)))
    assert prec == approx(0.25)


def test_precision_series_set():
    prec = _test_prec(pd.Series([1, 2, 3, 4]), [1, 3, 5])
    assert prec == approx(0.5)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), range(4, 10))
    assert prec == approx(0.25)


def test_precision_series_index():
    prec = _test_prec(pd.Series([1, 3]), pd.Index([1, 3]))
    assert prec == approx(1.0)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), pd.Index([1, 3, 5]))
    assert prec == approx(0.5)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), pd.Index(range(4, 10)))
    assert prec == approx(0.25)


def test_precision_series_array():
    prec = _test_prec(pd.Series([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), np.array([1, 3, 5]))
    assert prec == approx(0.5)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), np.arange(4, 10, 1, "u4"))
    assert prec == approx(0.25)


def test_precision_array():
    prec = _test_prec(np.array([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = _test_prec(np.array([1, 2, 3, 4]), np.array([1, 3, 5]))
    assert prec == approx(0.5)

    prec = _test_prec(np.array([1, 2, 3, 4]), np.arange(4, 10, 1, "u4"))
    assert prec == approx(0.25)


def test_prec_long_rel():
    rel = np.arange(100)
    items = [1, 0, 150, 3, 10]

    r = _test_prec(items, rel, k=5)
    assert r == approx(0.8)


def test_prec_long_items():
    rel = np.arange(100)
    items = [1, 0, 150, 3, 10, 30, 120, 4, 17]

    r = _test_prec(items, rel, k=5)
    assert r == approx(0.8)


def test_prec_short_items():
    rel = np.arange(100)
    items = [1, 0, 150]

    r = _test_prec(items, rel, k=5)
    assert r == approx(2 / 3)
