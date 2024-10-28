# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from pytest import approx, mark

from lenskit.data import ItemList
from lenskit.metrics.ranking import recip_rank
from lenskit.util.test import demo_recs  # noqa: F401


def _test_rr(items, rel, **kw):
    recs = ItemList(items, ordered=True)
    truth = ItemList(rel)
    return recip_rank(**kw)(recs, truth)


def test_mrr_empty_zero():
    rr = _test_rr([], [1, 3])
    assert rr == approx(0)


def test_mrr_norel_zero():
    "no relevant items -> zero"
    rr = _test_rr([1, 2, 3], [4, 5])
    assert rr == approx(0)


def test_mrr_first_one():
    "first relevant -> one"
    rr = _test_rr([1, 2, 3], [1, 4])
    assert rr == approx(1.0)


def test_mrr_second_one_half():
    "second relevant -> 0.5"
    rr = _test_rr([1, 2, 3], [5, 2, 3])
    assert rr == approx(0.5)


def test_mrr_series():
    "second relevant -> 0.5 in pd series"
    rr = _test_rr(pd.Series([1, 2, 3]), pd.Series([5, 2, 3]))
    assert rr == approx(0.5)


def test_mrr_series_idx():
    "second relevant -> 0.5 in pd series w/ index"
    rr = _test_rr(pd.Series([1, 2, 3]), pd.Index([5, 2, 3]))
    assert rr == approx(0.5)


def test_mrr_array_late():
    "deep -> 0.1"
    rr = _test_rr(np.arange(1, 21, 1, "u4"), [20, 10])
    assert rr == approx(0.1)


def test_mrr_k_trunc():
    rr = _test_rr(np.arange(1, 21, 1, "u4"), [20, 10], k=5)
    assert rr == approx(0.0)

    rr = _test_rr(np.arange(1, 21, 1, "u4"), [20, 10, 5], k=5)
    assert rr == approx(0.2)


def test_mrr_k_short():
    rr = _test_rr(np.arange(1, 5, 1, "u4"), [2], k=10)
    assert rr == approx(0.5)
