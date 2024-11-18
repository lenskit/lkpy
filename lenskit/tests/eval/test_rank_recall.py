# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd

from pytest import approx

from lenskit.data import ItemList
from lenskit.metrics.ranking import Recall
from lenskit.util.test import demo_recs  # noqa: F401

_log = logging.getLogger(__name__)


def _test_recall(items, rel, **kwargs):
    recs = ItemList(items, ordered=True)
    truth = ItemList(rel)
    return Recall(**kwargs).measure_list(recs, truth)


def test_recall_empty_zero():
    prec = _test_recall([], [1, 3])
    assert prec == approx(0)


def test_recall_norel_na():
    prec = _test_recall([1, 3], [])
    assert np.isnan(prec)


def test_recall_simple_cases():
    prec = _test_recall([1, 3], [1, 3])
    assert prec == approx(1.0)

    prec = _test_recall([1], [1, 3])
    assert prec == approx(0.5)

    prec = _test_recall([1, 2, 3, 4], [1, 3])
    assert prec == approx(1.0)

    prec = _test_recall([1, 2, 3, 4], [1, 3, 5])
    assert prec == approx(2.0 / 3)

    prec = _test_recall([1, 2, 3, 4], range(5, 10))
    assert prec == approx(0.0)

    prec = _test_recall([1, 2, 3, 4], range(4, 9))
    assert prec == approx(0.2)


def test_recall_series():
    prec = _test_recall(pd.Series([1, 3]), pd.Series([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(pd.Series([1, 2, 3]), pd.Series([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), pd.Series(range(4, 9)))
    assert prec == approx(0.2)


def test_recall_series_set():
    prec = _test_recall(pd.Series([1, 2, 3, 4]), [1, 3, 5, 7])
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), range(4, 9))
    assert prec == approx(0.2)


def test_recall_series_index():
    prec = _test_recall(pd.Series([1, 3]), pd.Index([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), pd.Index([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), pd.Index(range(4, 9)))
    assert prec == approx(0.2)


def test_recall_series_array():
    prec = _test_recall(pd.Series([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), np.array([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(pd.Series([1, 2, 3, 4]), np.arange(4, 9, 1, "u4"))
    assert prec == approx(0.2)


def test_recall_array():
    prec = _test_recall(np.array([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = _test_recall(np.array([1, 2, 3, 4]), np.array([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = _test_recall(np.array([1, 2, 3, 4]), np.arange(4, 9, 1, "u4"))
    assert prec == approx(0.2)


def test_recall_long_rel():
    rel = np.arange(100)
    items = [1, 0, 150, 3, 10]

    r = _test_recall(items, rel, k=5)
    assert r == approx(0.8)


def test_recall_long_items():
    rel = np.arange(100)
    items = [1, 0, 150, 3, 10, 30, 120, 4, 17]

    r = _test_recall(items, rel, k=5)
    assert r == approx(0.8)


def test_recall_partial_rel():
    rel = np.arange(100)
    items = [1, 0, 150, 3, 10]

    r = _test_recall(items, rel, k=10)
    assert r == approx(0.4)
