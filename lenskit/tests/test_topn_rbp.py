# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd

import hypothesis.strategies as st
from hypothesis import given
from pytest import approx, mark

from lenskit.metrics.topn import _bulk_rbp, rbp
from lenskit.topn import RecListAnalysis
from lenskit.util.test import demo_recs  # noqa: F401

_log = logging.getLogger(__name__)


def test_rbp_empty():
    recs = pd.DataFrame({"item": []})
    truth = pd.DataFrame({"item": [1, 2, 3], "rating": [3.0, 5.0, 4.0]})
    truth = truth.set_index("item")
    assert rbp(recs, truth) == approx(0.0)


def test_rbp_no_match():
    recs = pd.DataFrame({"item": [4]})
    truth = pd.DataFrame({"item": [1, 2, 3], "rating": [3.0, 5.0, 4.0]})
    truth = truth.set_index("item")
    assert rbp(recs, truth) == approx(0.0)


def test_rbp_one_match():
    recs = pd.DataFrame({"item": [1]})
    truth = pd.DataFrame({"item": [1, 2, 3], "rating": [3.0, 5.0, 4.0]})
    truth = truth.set_index("item")
    assert rbp(recs, truth) == approx(0.5)


@given(st.lists(st.integers(1), min_size=1, max_size=100, unique=True), st.floats(0.05, 0.95))
def test_rbp_perfect(items, p):
    n = len(items)
    recs = pd.DataFrame({"item": items})
    truth = pd.DataFrame({"item": items, "rating": 1})
    truth = truth.set_index("item").sort_index()
    assert rbp(recs, truth, patience=p) == approx(np.sum(p ** np.arange(n)) * (1 - p))


@given(st.lists(st.integers(1), min_size=1, max_size=100, unique=True), st.floats(0.05, 0.95))
def test_rbp_perfect_norm(items, p):
    recs = pd.DataFrame({"item": items})
    truth = pd.DataFrame({"item": items, "rating": 1})
    truth = truth.set_index("item").sort_index()
    assert rbp(recs, truth, patience=p, normalize=True) == approx(1.0)


@given(
    st.lists(st.integers(1), min_size=1, max_size=100, unique=True),
    st.integers(1, 100),
    st.floats(0.05, 0.95),
)
def test_rbp_perfect_k(items, k, p):
    n = len(items)
    eff_n = min(n, k)
    recs = pd.DataFrame({"item": items})
    truth = pd.DataFrame({"item": items, "rating": 1})
    truth = truth.set_index("item").sort_index()
    assert rbp(recs, truth, k=k, patience=p) == approx(np.sum(p ** np.arange(eff_n)) * (1 - p))


@given(
    st.lists(st.integers(1), min_size=1, max_size=100, unique=True),
    st.integers(1, 100),
    st.floats(0.05, 0.95),
)
def test_rbp_perfect_k_norm(items, k, p):
    recs = pd.DataFrame({"item": items})
    truth = pd.DataFrame({"item": items, "rating": 1})
    truth = truth.set_index("item").sort_index()
    assert rbp(recs, truth, k=k, patience=p, normalize=True) == approx(1.0)


def test_rbp_missing():
    recs = pd.DataFrame({"item": [1, 2]})
    truth = pd.DataFrame({"item": [1, 2, 3], "rating": [3.0, 5.0, 4.0]})
    truth = truth.set_index("item").sort_index()
    # (1 + 0.5) * 0.5
    assert rbp(recs, truth) == approx(0.75)


def test_rbp_bulk_at_top():
    truth = pd.DataFrame.from_records(
        [(1, 50, 3.5), (1, 30, 3.5)], columns=["LKTruthID", "item", "rating"]
    ).set_index(["LKTruthID", "item"])

    recs = pd.DataFrame.from_records(
        [(1, 1, 50, 1), (1, 1, 30, 2), (1, 1, 72, 3)],
        columns=["LKRecID", "LKTruthID", "item", "rank"],
    )

    rbp = _bulk_rbp(recs, truth)
    assert len(rbp) == 1
    assert rbp.index.tolist() == [1]
    assert rbp.iloc[0] == approx(0.75)


def test_rbp_bulk_not_at_top():
    truth = pd.DataFrame.from_records(
        [(1, 50, 3.5), (1, 30, 3.5)], columns=["LKTruthID", "item", "rating"]
    ).set_index(["LKTruthID", "item"])

    recs = pd.DataFrame.from_records(
        [(1, 1, 50, 1), (1, 1, 72, 2), (1, 1, 30, 3)],
        columns=["LKRecID", "LKTruthID", "item", "rank"],
    )

    rbp = _bulk_rbp(recs, truth)
    assert len(rbp) == 1
    assert rbp.index.tolist() == [1]
    assert rbp.iloc[0] == approx((1 + 0.25) * 0.5)


@mark.parametrize("normalize", [False, True])
def test_rbp_bulk_match(demo_recs, normalize):
    "bulk and normal match"
    train, test, recs = demo_recs

    rla = RecListAnalysis()
    rla.add_metric(rbp, normalize=normalize)
    rla.add_metric(rbp, name="rbp_k", k=5, normalize=normalize)
    # metric without the bulk capabilities
    rla.add_metric(lambda *a: rbp(*a, normalize=normalize), name="ind_rbp")
    rla.add_metric(lambda *a, **k: rbp(*a, normalize=normalize, **k), name="ind_rbp_k", k=5)
    res = rla.compute(recs, test)

    res["diff"] = np.abs(res.rbp - res.ind_rbp)
    rl = res.nlargest(5, "diff")
    _log.info("res:\n%s", rl)
    user = rl.index[0]
    _log.info("user: %s\n%s", user, rl.iloc[0])
    _log.info("test:\n%s", test[test["user"] == user])
    urecs = recs[recs["user"] == user].join(
        test.set_index(["user", "item"])["rating"], on=["user", "item"], how="left"
    )
    _log.info("recs:\n%s", urecs[urecs["rating"].notnull()])

    assert res.rbp.values == approx(res.ind_rbp.values)
    assert res.rbp_k.values == approx(res.ind_rbp_k.values)
