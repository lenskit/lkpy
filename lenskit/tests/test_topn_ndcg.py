# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from pytest import approx, mark

from lenskit.metrics.topn import _bulk_ndcg, _dcg, dcg, ndcg
from lenskit.topn import RecListAnalysis
from lenskit.util.test import demo_recs  # noqa: F401


def test_dcg_empty():
    "empty should be zero"
    assert _dcg(np.array([])) == approx(0)


def test_dcg_zeros():
    assert _dcg(np.zeros(10)) == approx(0)


def test_dcg_single():
    "a single element should be scored at the right place"
    assert _dcg(np.array([0.5])) == approx(0.5)
    assert _dcg(np.array([0, 0.5])) == approx(0.5)
    assert _dcg(np.array([0, 0, 0.5])) == approx(0.5 / np.log2(3))
    assert _dcg(np.array([0, 0, 0.5, 0])) == approx(0.5 / np.log2(3))


def test_dcg_mult():
    "multiple elements should score correctly"
    assert _dcg(np.array([np.e, np.pi])) == approx(np.e + np.pi)
    assert _dcg(np.array([np.e, 0, 0, np.pi])) == approx(np.e + np.pi / np.log2(4))


def test_dcg_empty2():
    "empty should be zero"
    assert _dcg(np.array([])) == approx(0)


def test_dcg_zeros2():
    assert _dcg(np.zeros(10)) == approx(0)


def test_dcg_single2():
    "a single element should be scored at the right place"
    assert _dcg(np.array([0.5])) == approx(0.5)
    assert _dcg(np.array([0, 0.5])) == approx(0.5)
    assert _dcg(np.array([0, 0, 0.5])) == approx(0.5 / np.log2(3))
    assert _dcg(np.array([0, 0, 0.5, 0])) == approx(0.5 / np.log2(3))


def test_dcg_nan():
    "NANs should be 0"
    assert _dcg(np.array([np.nan, 0.5])) == approx(0.5)


def test_dcg_series():
    "The DCG function should work on a series"
    assert _dcg(pd.Series([np.e, 0, 0, np.pi])) == approx((np.e + np.pi / np.log2(4)))


def test_dcg_mult2():
    "multiple elements should score correctly"
    assert _dcg(np.array([np.e, np.pi])) == approx(np.e + np.pi)
    assert _dcg(np.array([np.e, 0, 0, np.pi])) == approx((np.e + np.pi / np.log2(4)))


def test_ndcg_empty():
    recs = pd.DataFrame({"item": []})
    truth = pd.DataFrame({"item": [1, 2, 3], "rating": [3.0, 5.0, 4.0]})
    truth = truth.set_index("item")
    assert ndcg(recs, truth) == approx(0.0)


def test_ndcg_no_match():
    recs = pd.DataFrame({"item": [4]})
    truth = pd.DataFrame({"item": [1, 2, 3], "rating": [3.0, 5.0, 4.0]})
    truth = truth.set_index("item")
    assert ndcg(recs, truth) == approx(0.0)


def test_ndcg_perfect():
    recs = pd.DataFrame({"item": [2, 3, 1]})
    truth = pd.DataFrame({"item": [1, 2, 3], "rating": [3.0, 5.0, 4.0]})
    truth = truth.set_index("item")
    assert ndcg(recs, truth) == approx(1.0)


def test_ndcg_perfect_k_short():
    recs = pd.DataFrame({"item": [2, 3, 1]})
    truth = pd.DataFrame({"item": [1, 2, 3], "rating": [3.0, 5.0, 4.0]})
    truth = truth.set_index("item")
    assert ndcg(recs, truth, k=2) == approx(1.0)
    assert ndcg(recs[:2], truth, k=2) == approx(1.0)


def test_ndcg_wrong():
    recs = pd.DataFrame({"item": [1, 2]})
    truth = pd.DataFrame({"item": [1, 2, 3], "rating": [3.0, 5.0, 4.0]})
    truth = truth.set_index("item")
    assert ndcg(recs, truth) == approx(_dcg([3.0, 5.0] / _dcg([5.0, 4.0, 3.0])))


def test_ndcg_perfect_k():
    recs = pd.DataFrame({"item": [2, 3]})
    truth = pd.DataFrame({"item": [1, 2, 3], "rating": [3.0, 5.0, 4.0]})
    truth = truth.set_index("item")
    assert ndcg(recs, truth, k=2) == approx(1.0)


def test_ndcg_bulk_at_top():
    truth = pd.DataFrame.from_records(
        [(1, 50, 3.5), (1, 30, 3.5)], columns=["LKTruthID", "item", "rating"]
    ).set_index(["LKTruthID", "item"])

    recs = pd.DataFrame.from_records(
        [(1, 1, 50, 1), (1, 1, 30, 2), (1, 1, 72, 3)],
        columns=["LKRecID", "LKTruthID", "item", "rank"],
    )

    ndcg = _bulk_ndcg(recs, truth)
    assert len(ndcg) == 1
    assert ndcg.index.tolist() == [1]
    assert ndcg.iloc[0] == approx(1.0)


def test_ndcg_bulk_not_at_top():
    truth = pd.DataFrame.from_records(
        [(1, 50, 3.5), (1, 30, 3.5)], columns=["LKTruthID", "item", "rating"]
    ).set_index(["LKTruthID", "item"])

    recs = pd.DataFrame.from_records(
        [(1, 1, 50, 1), (1, 1, 72, 2), (1, 1, 30, 3)],
        columns=["LKRecID", "LKTruthID", "item", "rank"],
    )

    ndcg = _bulk_ndcg(recs, truth)
    assert len(ndcg) == 1
    assert ndcg.index.tolist() == [1]
    assert ndcg.iloc[0] == approx(0.8155, abs=0.001)


@mark.parametrize("drop_rating", [False, True])
def test_ndcg_bulk_match(demo_recs, drop_rating):
    "bulk and normal match"
    train, test, recs = demo_recs
    if drop_rating:
        test = test[["user", "item"]]

    rla = RecListAnalysis()
    rla.add_metric(ndcg)
    rla.add_metric(ndcg, name="ndcg_k", k=5)
    rla.add_metric(dcg)
    # metric without the bulk capabilities
    rla.add_metric(lambda *a: ndcg(*a), name="ind_ndcg")
    rla.add_metric(lambda *a, **k: ndcg(*a, **k), name="ind_ndcg_k", k=5)
    res = rla.compute(recs, test)

    res["ind_ideal"] = res["dcg"] / res["ind_ndcg"]
    print(res)

    assert res.ndcg.values == approx(res.ind_ndcg.values)
