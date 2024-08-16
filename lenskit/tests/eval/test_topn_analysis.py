# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pytest import approx, mark

from lenskit import batch, topn
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender
from lenskit.algorithms.knn.item import ItemItem
from lenskit.algorithms.knn.user import UserUser
from lenskit.data import from_interactions_df
from lenskit.metrics.topn import _dcg, precision, recall
from lenskit.util.test import demo_recs  # noqa: F401

_log = logging.getLogger(__name__)


def test_split_keys():
    recs, truth = topn._df_keys(
        ["algorithm", "user", "item", "rank", "score"], ["user", "item", "rating"]
    )
    assert truth == ["user"]
    assert recs == ["algorithm", "user"]


def test_split_keys_gcol():
    recs, truth = topn._df_keys(
        ["algorithm", "user", "item", "rank", "score", "fishtank"],
        ["user", "item", "rating"],
        ["algorithm", "fishtank", "user"],
    )
    assert truth == ["user"]
    assert recs == ["algorithm", "fishtank", "user"]


def test_run_one():
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)

    recs = pd.DataFrame({"user": 1, "item": [2]})
    recs.name = "recs"
    truth = pd.DataFrame({"user": 1, "item": [1, 2, 3], "rating": [3.0, 5.0, 4.0]})
    truth.name = "truth"

    print(recs)
    print(truth)

    res = rla.compute(recs, truth)
    print(res)

    assert res.index.name == "user"
    assert res.index.is_unique

    assert len(res) == 1
    assert all(res.index == 1)
    assert all(res.precision == 1.0)
    assert res.recall.values == approx(1 / 3)


def test_run_two():
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)
    rla.add_metric(topn.ndcg)

    recs = pd.DataFrame(
        {
            "data": "a",
            "user": ["a", "a", "a", "b", "b"],
            "item": [2, 3, 1, 4, 5],
            "rank": [1, 2, 3, 1, 2],
        }
    )
    truth = pd.DataFrame(
        {
            "user": ["a", "a", "a", "b", "b", "b"],
            "item": [1, 2, 3, 1, 5, 6],
            "rating": [3.0, 5.0, 4.0, 3.0, 5.0, 4.0],
        }
    )

    def prog(inner):
        assert len(inner) == 2
        return inner

    res = rla.compute(recs, truth)
    print(res)

    assert res.columns.nlevels == 1
    assert len(res) == 2
    assert res.index.nlevels == 2
    assert res.index.names == ["data", "user"]
    assert all(res.index.levels[0] == "a")
    assert all(res.index.levels[1] == ["a", "b"])
    assert all(res.reset_index().user == ["a", "b"])
    partial_ndcg = _dcg([0.0, 5.0]) / _dcg([5, 4, 3])
    assert res.ndcg.values == approx([1.0, partial_ndcg])
    assert res.precision.values == approx([1.0, 1 / 2])
    assert res.recall.values == approx([1.0, 1 / 3])


def test_inner_format():
    rla = topn.RecListAnalysis()

    recs = pd.DataFrame(
        {
            "data": "a",
            "user": ["a", "a", "a", "b", "b"],
            "item": [2, 3, 1, 4, 5],
            "rank": [1, 2, 3, 1, 2],
        }
    )
    truth = pd.DataFrame(
        {
            "user": ["a", "a", "a", "b", "b", "b"],
            "item": [1, 2, 3, 1, 5, 6],
            "rating": [3.0, 5.0, 4.0, 3.0, 5.0, 4.0],
        }
    )

    def inner(recs, truth, foo="a"):
        assert foo == "b"
        assert set(recs.columns) == set(["LKRecID", "LKTruthID", "item", "rank"])
        assert truth.index.name == "item"
        assert truth.index.is_unique
        print(truth)
        assert all(truth.columns == ["rating"])
        return len(recs.join(truth, on="item", how="inner"))

    rla.add_metric(inner, name="bob", foo="b")

    res = rla.compute(recs, truth)
    print(res)

    assert len(res) == 2
    assert res.index.nlevels == 2
    assert res.index.names == ["data", "user"]
    assert all(res.index.levels[0] == "a")
    assert all(res.index.levels[1] == ["a", "b"])
    assert all(res.reset_index().user == ["a", "b"])
    assert all(res["bob"] == [3, 1])


def test_spec_group_cols():
    rla = topn.RecListAnalysis(group_cols=["data", "user"])
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)
    rla.add_metric(topn.ndcg)

    recs = pd.DataFrame(
        {
            "data": "a",
            "user": ["a", "a", "a", "b", "b"],
            "item": [2, 3, 1, 4, 5],
            "rank": [1, 2, 3, 1, 2],
            "wombat": np.random.randn(5),
        }
    )
    truth = pd.DataFrame(
        {
            "user": ["a", "a", "a", "b", "b", "b"],
            "item": [1, 2, 3, 1, 5, 6],
            "rating": [3.0, 5.0, 4.0, 3.0, 5.0, 4.0],
        }
    )

    res = rla.compute(recs, truth)
    print(res)

    assert len(res) == 2
    assert res.index.nlevels == 2
    assert res.index.names == ["data", "user"]
    assert all(res.index.levels[0] == "a")
    assert all(res.index.levels[1] == ["a", "b"])
    assert all(res.reset_index().user == ["a", "b"])
    partial_ndcg = _dcg([0.0, 5.0]) / _dcg([5, 4, 3])
    assert res.ndcg.values == approx([1.0, partial_ndcg])
    assert res.precision.values == approx([1.0, 1 / 2])
    assert res.recall.values == approx([1.0, 1 / 3])


def test_java_equiv():
    dir = Path(__file__).parent
    metrics = pd.read_csv(str(dir / "topn-java-metrics.csv"))
    recs = pd.read_csv(str(dir / "topn-java-recs.csv"))
    truth = pd.read_csv(str(dir / "topn-java-truth.csv"))

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    res = rla.compute(recs, truth)

    umm = pd.merge(metrics, res.reset_index())
    umm["err"] = umm["ndcg"] - umm["Java.nDCG"]
    _log.info("merged: \n%s", umm)
    assert umm["err"].values == approx(0, abs=1.0e-6)


@mark.parametrize("drop_rating", [False, True])
def test_pr_bulk_match(demo_recs, drop_rating):
    "bulk and normal match"
    train, test, recs = demo_recs
    if drop_rating:
        test = test[["user", "item"]]

    _log.info("creating analysis")
    rla = topn.RecListAnalysis()
    rla.add_metric(precision)
    rla.add_metric(recall)
    # metric without the bulk capabilities
    rla.add_metric(lambda *a: precision(*a), name="ind_p")
    rla.add_metric(lambda *a: recall(*a), name="ind_r")
    res = rla.compute(recs, test)

    print(res)
    _log.info("precision mismatches:\n%s", res[res.precision != res.ind_p])
    _log.info("recall mismatches:\n%s", res[res.recall != res.ind_r])

    assert res.precision.values == approx(res.ind_p.values)
    assert res.recall.values == approx(res.ind_r.values)
