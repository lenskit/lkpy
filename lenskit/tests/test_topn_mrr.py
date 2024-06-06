# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from pytest import approx, mark

from lenskit.topn import RecListAnalysis, recip_rank
from lenskit.util.test import demo_recs  # noqa: F401


def _test_rr(items, rel, **kw):
    recs = pd.DataFrame({"item": items})
    truth = pd.DataFrame({"item": rel}).set_index("item")
    return recip_rank(recs, truth, **kw)


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


@mark.parametrize("drop_rating", [False, True])
def test_mrr_bulk(demo_recs, drop_rating):
    "bulk and normal match"
    train, test, recs = demo_recs
    if drop_rating:
        test = test[["user", "item"]]

    rla = RecListAnalysis()
    rla.add_metric(recip_rank)
    rla.add_metric(recip_rank, name="rr_k", k=10)
    # metric without the bulk capabilities
    rla.add_metric(lambda *a: recip_rank(*a), name="ind_rr")
    rla.add_metric(lambda *a, **k: recip_rank(*a, **k), name="ind_rr_k", k=10)
    res = rla.compute(recs, test)

    assert all(res.recip_rank == res.ind_rr)
    assert all(res.rr_k == res.ind_rr_k)
