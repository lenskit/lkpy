# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

import pytest

import lenskit.crossfold as xf
import lenskit.util.test as lktu
from lenskit import batch, topn
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import PopScore, TopN
from lenskit.algorithms.bias import Bias
from lenskit.data import Dataset, from_interactions_df

_log = logging.getLogger(__name__)


class MLB(NamedTuple):
    ratings: pd.DataFrame
    data: Dataset
    algo: TopN


@pytest.fixture
def mlb(ml_ratings, ml_ds) -> MLB:
    algo = TopN(Bias())
    algo.fit(ml_ds)
    return MLB(
        ml_ratings.rename(
            columns={
                "userId": "user",
                "movieId": "item",
            }
        ),
        ml_ds,
        algo,
    )


class MLFolds:
    def __init__(self, ratings):
        self.ratings = ratings
        self.folds = list(xf.partition_users(self.ratings, 5, xf.SampleFrac(0.2)))
        self.test = pd.concat(f.test for f in self.folds)

    def evaluate(self, algo, train, test, **kwargs):
        _log.info("running training")
        algo.fit(from_interactions_df(train))
        _log.info("testing %d users", test.user.nunique())
        recs = batch.recommend(algo, test.user.unique(), 100, **kwargs)
        return recs

    def eval_all(self, algo, **kwargs):
        return pd.concat(self.evaluate(algo, train, test, **kwargs) for (train, test) in self.folds)

    def check_positive_ndcg(self, recs):
        _log.info("analyzing recommendations")
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.ndcg)
        results = rla.compute(recs, self.test)
        dcg = results.ndcg
        _log.info("nDCG for %d users is %f (max=%f)", len(dcg), dcg.mean(), dcg.max())
        assert dcg.mean() > 0


@pytest.fixture
def ml_folds(ml_100k) -> MLFolds:
    return MLFolds(ml_100k)


def test_recommend_single(mlb: MLB):
    res = batch.recommend(mlb.algo, [1], None, {1: [31]}, n_jobs=1)

    assert len(res) == 1
    assert all(res["user"] == 1)
    assert all(res["rank"] == 1)
    assert set(res.columns) == set(["user", "rank", "item", "score"])

    algo = mlb.algo.predictor
    expected = algo.mean_ + algo.item_offsets_.loc[31] + algo.user_offsets_.loc[1]
    assert res.score.iloc[0] == pytest.approx(expected)


def test_recommend_user(mlb: MLB):
    uid = 5
    items = mlb.ratings.item.unique()

    def candidates(user):
        urs = mlb.ratings[mlb.ratings.user == user]
        return np.setdiff1d(items, urs.item.unique())

    res = batch.recommend(mlb.algo, [5], 10, candidates, n_jobs=1)

    assert len(res) == 10
    assert set(res.columns) == set(["user", "rank", "item", "score"])
    assert all(res["user"] == uid)
    assert all(res["rank"] == np.arange(10) + 1)
    # they should be in decreasing order
    assert all(np.diff(res.score) <= 0)


def test_recommend_two_users(mlb: MLB):
    items = mlb.ratings.item.unique()

    def candidates(user):
        urs = mlb.ratings[mlb.ratings.user == user]
        return np.setdiff1d(items, urs.item.unique())

    res = batch.recommend(mlb.algo, [5, 10], 10, candidates, n_jobs=1)

    assert len(res) == 20
    assert set(res.user) == set([5, 10])
    assert all(res.groupby("user").item.count() == 10)
    assert all(res.groupby("user")["rank"].max() == 10)
    assert all(np.diff(res[res.user == 5].score) <= 0)
    assert all(np.diff(res[res.user == 5]["rank"]) == 1)
    assert all(np.diff(res[res.user == 10].score) <= 0)
    assert all(np.diff(res[res.user == 10]["rank"]) == 1)


def test_recommend_no_cands(mlb: MLB):
    res = batch.recommend(mlb.algo, [5, 10], 10, n_jobs=1)

    assert len(res) == 20
    assert set(res.user) == set([5, 10])
    assert all(res.groupby("user").item.count() == 10)
    assert all(res.groupby("user")["rank"].max() == 10)
    assert all(np.diff(res[res.user == 5].score) <= 0)
    assert all(np.diff(res[res.user == 5]["rank"]) == 1)
    assert all(np.diff(res[res.user == 10].score) <= 0)
    assert all(np.diff(res[res.user == 10]["rank"]) == 1)

    idx_rates = mlb.ratings.set_index(["user", "item"])
    merged = res.join(idx_rates, on=["user", "item"], how="inner")
    assert len(merged) == 0


@pytest.mark.parametrize(("ncpus"), [None, 1, 2])
@pytest.mark.eval
def test_bias_batch_recommend(ml_folds: MLFolds, ncpus):
    algo = Bias(damping=5)
    algo = TopN(algo)

    recs = ml_folds.eval_all(algo, n_jobs=ncpus)

    ml_folds.check_positive_ndcg(recs)


@pytest.mark.parametrize("ncpus", [None, 1, 2])
@pytest.mark.eval
def test_pop_batch_recommend(ml_folds: MLFolds, ncpus):
    algo = PopScore()
    algo = Recommender.adapt(algo)

    recs = ml_folds.eval_all(algo, n_jobs=ncpus)
    ml_folds.check_positive_ndcg(recs)
