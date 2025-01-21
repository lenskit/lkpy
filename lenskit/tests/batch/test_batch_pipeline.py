# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
from typing import Generator, NamedTuple

import numpy as np
import pandas as pd

from pytest import approx, fixture, mark

from lenskit.basic import BiasScorer, PopScorer
from lenskit.batch import BatchPipelineRunner, predict, recommend, score
from lenskit.data import Dataset, ItemList, UserIDKey, from_interactions_df
from lenskit.data.adapt import normalize_interactions_df
from lenskit.metrics import NDCG, RBP, RMSE, RunAnalysis
from lenskit.pipeline import Pipeline, topn_pipeline
from lenskit.splitting import SampleN, TTSplit, sample_users
from lenskit.testing import ml_100k, ml_ds, ml_ratings  # noqa: F401

_log = logging.getLogger(__name__)


class MLB(NamedTuple):
    ratings: pd.DataFrame
    data: Dataset
    pipeline: Pipeline


@fixture
def mlb(ml_ratings: pd.DataFrame, ml_ds: Dataset) -> MLB:
    bias = BiasScorer()
    pipeline = topn_pipeline(bias, predicts_ratings=True)
    pipeline.train(ml_ds)

    return MLB(
        normalize_interactions_df(ml_ratings),
        ml_ds,
        pipeline,
    )


@fixture
def ml_split(ml_100k: pd.DataFrame) -> Generator[TTSplit, None, None]:
    ds = from_interactions_df(ml_100k)
    yield sample_users(ds, 200, SampleN(5))


def test_predict_single(mlb: MLB):
    res = predict(mlb.pipeline, {1: ItemList([31])}, n_jobs=1)

    assert len(res) == 1
    uid, result = next(iter(res))
    assert isinstance(uid, UserIDKey)
    assert uid.user_id == 1
    assert len(result) == 1
    assert result.ids()[0] == 31

    preds = result.field("score")
    assert preds is not None
    assert preds >= 1 and preds <= 5


def test_score_single(mlb: MLB):
    res = score(mlb.pipeline, {1: ItemList([31])}, n_jobs=1)

    assert len(res) == 1
    uid, result = next(iter(res))
    assert isinstance(uid, UserIDKey)
    assert uid.user_id == 1
    assert len(result) == 1
    assert result.ids()[0] == 31

    preds = result.field("score")
    assert preds is not None
    assert preds >= 1 and preds <= 5


def test_recommend_user(mlb: MLB):
    user = 5

    results = recommend(mlb.pipeline, [user], n=10, n_jobs=1)

    assert len(results) == 1

    uid, ranking = next(iter(results))
    assert isinstance(uid, UserIDKey)
    assert uid.user_id == user
    assert isinstance(ranking, ItemList)
    assert ranking.ordered
    assert len(ranking) == 10

    # they should be in decreasing order
    score = ranking.scores()
    assert score is not None
    assert all(np.diff(score) <= 0)


@mark.parametrize(("ncpus"), [None, 1, 2])
@mark.eval
def test_bias_batch(ml_split: TTSplit, ncpus: int | None):
    algo = BiasScorer(damping=5)
    pipeline = topn_pipeline(algo, predicts_ratings=True, n=20)
    pipeline.train(ml_split.train)

    runner = BatchPipelineRunner(n_jobs=ncpus)
    runner.recommend()
    runner.predict()

    results = runner.run(pipeline, ml_split.test)

    preds = results.output("predictions")

    pa = RunAnalysis()
    pa.add_metric(RMSE())
    pred_acc = pa.compute(preds, ml_split.test)
    pas = pred_acc.list_summary()
    print(pas)
    assert pas.loc["RMSE", "mean"] == approx(0.949, rel=0.1)

    recs = results.output("recommendations")
    ra = RunAnalysis()
    ra.add_metric(NDCG())
    ra.add_metric(RBP())
    rec_acc = ra.compute(recs, ml_split.test)
    ras = rec_acc.list_summary()
    print(ras)

    assert ras.loc["RBP", "mean"] > 0
    assert ras.loc["NDCG", "mean"] > 0


@mark.parametrize("ncpus", [None, 1, 2])
@mark.eval
def test_pop_batch_recommend(ml_split: TTSplit, ncpus: int | None):
    algo = PopScorer()
    pipeline = topn_pipeline(algo, predicts_ratings=True, n=20)
    pipeline.train(ml_split.train)

    runner = BatchPipelineRunner(n_jobs=ncpus)
    runner.recommend()

    results = runner.run(pipeline, ml_split.test)

    recs = results.output("recommendations")
    ra = RunAnalysis()
    ra.add_metric(NDCG())
    ra.add_metric(RBP())
    rec_acc = ra.compute(recs, ml_split.test)
    ras = rec_acc.list_summary()
    print(ras)

    assert ras.loc["RBP", "mean"] > 0
    assert ras.loc["NDCG", "mean"] > 0
