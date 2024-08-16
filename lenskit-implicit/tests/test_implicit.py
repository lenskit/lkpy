# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd

from pytest import mark

import lenskit.util.test as lktu
from lenskit import util
from lenskit.data import from_interactions_df
from lenskit.implicit import ALS, BPR

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)


@mark.slow
def test_implicit_als_train_rec(ml_ds):
    algo = ALS(25)
    assert algo.factors == 25

    ret = algo.fit(ml_ds)
    assert ret is algo

    recs = algo.recommend(100, n=20)
    assert len(recs) == 20

    preds = algo.predict_for_user(100, items=[1, 377])
    assert np.all(preds.notnull())

    _log.info("serializing implicit model")
    mod = pickle.dumps(algo)
    _log.info("serialized to %d bytes")
    a2 = pickle.loads(mod)

    r2 = a2.recommend(100, n=20)
    assert len(r2) == 20
    assert all(r2 == recs)


@mark.slow
@mark.eval
@mark.parametrize("n_jobs", [1, None])
def test_implicit_als_batch_accuracy(ml_100k, n_jobs):
    import lenskit.crossfold as xf
    from lenskit import batch, topn

    algo_t = ALS(25)

    def eval(train, test):
        _log.info("running training")
        train["rating"] = train.rating.astype(np.float_)
        algo = util.clone(algo_t)
        algo.fit(from_interactions_df(train))
        users = test.user.unique()
        _log.info("testing %d users", len(users))
        recs = batch.recommend(algo, users, 100, n_jobs=n_jobs)
        return recs

    folds = list(xf.partition_users(ml_100k, 5, xf.SampleFrac(0.2)))
    test = pd.concat(f.test for f in folds)

    recs = pd.concat(eval(train, test) for (train, test) in folds)

    _log.info("analyzing recommendations")
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, test)
    dcg = results.ndcg
    _log.info("nDCG for %d users is %.4f", len(dcg), dcg.mean())
    assert dcg.mean() > 0


@mark.slow
def test_implicit_bpr_train_rec(ml_ds):
    algo = BPR(25, use_gpu=False)
    assert algo.factors == 25

    algo.fit(ml_ds)

    recs = algo.recommend(100, n=20)
    assert len(recs) == 20

    preds = algo.predict_for_user(100, [20, 30, 23148010])
    assert all(preds.index == [20, 30, 23148010])
    assert all(preds.isna() == [False, False, True])

    _log.info("serializing implicit model")
    mod = pickle.dumps(algo)
    _log.info("serialized to %d bytes")
    a2 = pickle.loads(mod)

    r2 = a2.recommend(100, n=20)
    assert len(r2) == 20
    assert all(r2 == recs)


def test_implicit_pickle_untrained(tmp_path):
    mf = tmp_path / "bpr.dat"
    algo = BPR(25, use_gpu=False)

    with mf.open("wb") as f:
        pickle.dump(algo, f)

    with mf.open("rb") as f:
        a2 = pickle.load(f)

    assert a2 is not algo
    assert a2.factors == 25
