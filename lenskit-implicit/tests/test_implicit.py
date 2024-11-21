# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np

from pytest import approx, mark

import lenskit.util.test as lktu
from lenskit import util
from lenskit.data import ItemList, from_interactions_df
from lenskit.implicit import ALS, BPR
from lenskit.metrics import quick_measure_model

_log = logging.getLogger(__name__)


@mark.slow
def test_implicit_als_train_rec(ml_ds):
    algo = ALS(25)
    assert algo.factors == 25

    ret = algo.train(ml_ds)
    assert ret is algo

    preds = algo(100, items=ItemList([1, 377]))
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert np.all(preds.notna())

    _log.info("serializing implicit model")
    mod = pickle.dumps(algo)
    _log.info("serialized to %d bytes")
    a2 = pickle.loads(mod)

    p2 = a2(100, items=ItemList([1, 377])).scores("pandas", index="ids")
    assert np.all(p2 == preds)


@mark.slow
@mark.eval
@mark.parametrize("n_jobs", [1, None])
def test_implicit_als_batch_accuracy(ml_100k, n_jobs):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(ALS(25), ds, n_jobs=n_jobs)

    ndcg = results.list_summary().loc["NDCG", "mean"]
    _log.info("nDCG for %d users is %.4f", len(results.list_metrics()), ndcg)
    assert ndcg > 0


@mark.slow
def test_implicit_bpr_train_rec(ml_ds):
    algo = BPR(25, use_gpu=False)
    assert algo.factors == 25

    algo.train(ml_ds)

    preds = algo(100, ItemList([20, 30, 23148010]))
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert all(preds.index == [20, 30, 23148010])
    assert all(preds.isna() == [False, False, True])

    _log.info("serializing implicit model")
    mod = pickle.dumps(algo)
    _log.info("serialized to %d bytes")
    a2 = pickle.loads(mod)

    p2 = a2(100, items=ItemList([20, 30, 23148010])).scores("pandas", index="ids")
    assert p2.values == approx(preds.values, nan_ok=True)


@mark.slow
@mark.eval
@mark.parametrize("n_jobs", [1, None])
def test_implicit_bpr_batch_accuracy(ml_100k, n_jobs):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(BPR(25), ds, n_jobs=n_jobs)

    ndcg = results.list_summary().loc["NDCG", "mean"]
    _log.info("nDCG for %d users is %.4f", len(results.list_metrics()), ndcg)
    assert ndcg > 0


def test_implicit_pickle_untrained(tmp_path):
    mf = tmp_path / "bpr.dat"
    algo = BPR(25, use_gpu=False)

    with mf.open("wb") as f:
        pickle.dump(algo, f)

    with mf.open("rb") as f:
        a2 = pickle.load(f)

    assert a2 is not algo
    assert a2.factors == 25
