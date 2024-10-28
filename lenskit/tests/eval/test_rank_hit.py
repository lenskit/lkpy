# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd

from pytest import approx

from lenskit.metrics.ranking import hit
from lenskit.util.test import demo_recs  # noqa: F401

_log = logging.getLogger(__name__)


def _test_hit(items, rel, **kwargs):
    recs = pd.DataFrame({"item": items})
    truth = pd.DataFrame({"item": rel}).set_index("item")
    return hit(recs, truth, **kwargs)


def test_hit_empty_zero():
    hr = _test_hit([], [1, 3])
    assert hr == 0


def test_hit_norel_na():
    hr = _test_hit([1, 3], [])
    assert hr is None


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

    r = _test_hit(items, rel, k=5)
    assert r == 1

    items += 200
    items[5] = 5

    r = _test_hit(np.array(items) + 200, rel, k=5)
    assert r == 0


def test_hit_partial_rel():
    rel = np.arange(100)
    items = [1, 0, 150, 3, 10]

    r = _test_hit(items, rel, k=10)
    assert r == 1


def test_hit_bulk_k(demo_recs):
    "bulk and normal match"
    train, test, recs = demo_recs
    assert test["user"].value_counts().max() > 5

    rla = topn.RecListAnalysis()
    rla.add_metric(hit, name="hk", k=5)
    rla.add_metric(hit)
    # metric without the bulk capabilities
    rla.add_metric(lambda *a, **k: hit(*a, **k), name="ind_hk", k=5)
    rla.add_metric(lambda *a: hit(*a), name="ind_h")
    res = rla.compute(recs, test)

    print(res)
    _log.info("recall mismatches:\n%s", res[res.hit != res.ind_h])

    assert res.hit.values == approx(res.ind_h.values)
    assert res.hk.values == approx(res.ind_hk.values)
