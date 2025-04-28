# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from pytest import approx, fail, mark

from lenskit import recommend
from lenskit.data import (
    Dataset,
    ItemList,
    ItemListCollection,
    RecQuery,
    UserIDKey,
    from_interactions_df,
)
from lenskit.knn import UserKNNScorer
from lenskit.metrics import call_metric, quick_measure_model
from lenskit.pipeline.common import predict_pipeline, topn_pipeline
from lenskit.testing import BasicComponentTests, ScorerTests
from lenskit.torch import inference_mode

_log = logging.getLogger(__name__)


class TestUserKNN(BasicComponentTests, ScorerTests):
    can_score = "some"
    component = UserKNNScorer


def test_uu_train(ml_ratings, ml_ds):
    algo = UserKNNScorer(k=30)
    algo.train(ml_ds)

    # we have data structures
    assert algo.user_means is not None
    assert algo.user_vectors is not None
    assert algo.user_ratings is not None

    # it should have computed correct means
    u_stats = ml_ds.user_stats()
    mlmeans = pd.Series(algo.user_means, index=algo.users.ids(), name="mean")
    mlmeans.index.name = "user_id"
    umeans, mlmeans = u_stats["mean_rating"].align(mlmeans)
    assert mlmeans.values == approx(umeans.values)

    # we should be able to reconstruct rating values
    uir = ml_ratings.set_index(["user_id", "item_id"]).rating
    rates = algo.user_ratings.to_scipy().tocoo()
    ui_rbdf = pd.DataFrame(
        {
            "user_id": algo.users.ids(rates.row),
            "item_id": algo.items.ids(rates.col),
            "nrating": rates.data,
        }
    ).set_index(["user_id", "item_id"])
    ui_rbdf = ui_rbdf.join(mlmeans)
    ui_rbdf["rating"] = ui_rbdf["nrating"] + ui_rbdf["mean"]
    ui_rbdf["orig_rating"] = uir
    assert ui_rbdf.rating.values == approx(ui_rbdf.orig_rating.values)


def test_uu_predict_one(ml_ds):
    algo = UserKNNScorer(k=30)
    algo.train(ml_ds)

    preds = algo(query=4, items=ItemList([1016]))
    assert len(preds) == 1
    assert preds.ids() == [1016]
    assert preds.scores() == approx([3.62221550680778])


def test_uu_predict_too_few(ml_ds):
    algo = UserKNNScorer(k=30, min_nbrs=2)
    algo.train(ml_ds)

    preds = algo(query=4, items=ItemList([2091]))
    assert len(preds) == 1
    assert preds.ids() == [2091]
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert all(preds.isna())


def test_uu_predict_too_few_blended(ml_ds):
    algo = UserKNNScorer(k=30, min_nbrs=2)
    algo.train(ml_ds)

    preds = algo(query=4, items=ItemList([1016, 2091]))
    assert len(preds) == 2
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert np.isnan(preds.loc[2091])
    assert preds.loc[1016] == approx(3.62221550680778)


def test_uu_predict_live_ratings(ml_ratings):
    algo = UserKNNScorer(k=30, min_nbrs=2)
    no4 = ml_ratings[ml_ratings.user_id != 4]
    no4 = from_interactions_df(no4)
    algo.train(no4)

    ratings = ItemList.from_df(ml_ratings[ml_ratings.user_id == 4][["item_id", "rating"]])

    query = RecQuery(20381, ratings)
    preds = algo(
        query=query,
        items=ItemList([1016, 2091]),
    )
    assert len(preds) == 2
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert np.isnan(preds.loc[2091])
    assert preds.loc[1016] == approx(3.62221550680778)


def test_uu_save_load(tmp_path, ml_ratings, ml_ds):
    orig = UserKNNScorer(k=30)
    _log.info("training model")
    orig.train(ml_ds)

    fn = tmp_path / "uu.model"
    _log.info("saving to %s", fn)
    with fn.open("wb") as f:
        pickle.dump(orig, f)

    _log.info("reloading model")
    with fn.open("rb") as f:
        algo = pickle.load(f)

    _log.info("checking model")

    # it should have computed correct means
    umeans = ml_ds.user_stats()["mean_rating"]
    mlmeans = pd.Series(algo.user_means, index=algo.users, name="mean")
    mlmeans.index.name = "user_id"
    umeans, mlmeans = umeans.align(mlmeans)
    assert mlmeans.values == approx(umeans.values)

    # we should be able to reconstruct rating values
    uir = ml_ratings.set_index(["user_id", "item_id"]).rating
    rates = algo.user_ratings.to_scipy().tocoo()
    ui_rbdf = pd.DataFrame(
        {
            "user_id": algo.users.ids(rates.row),
            "item_id": algo.items.ids(rates.col),
            "nrating": rates.data,
        }
    ).set_index(["user_id", "item_id"])
    ui_rbdf = ui_rbdf.join(mlmeans)
    ui_rbdf["rating"] = ui_rbdf["nrating"] + ui_rbdf["mean"]
    ui_rbdf["orig_rating"] = uir
    assert ui_rbdf.rating.values == approx(ui_rbdf.orig_rating.values)

    # running the predictor should work
    preds = algo(query=4, items=ItemList([1016]))
    assert len(preds) == 1
    assert preds.ids() == [1016]
    assert preds.scores() == approx([3.62221550680778])


def test_uu_predict_unknown_empty(ml_ds):
    algo = UserKNNScorer(k=30, min_nbrs=2)
    algo.train(ml_ds)

    preds = algo(query=-28018, items=ItemList([1016, 2091]))
    assert len(preds) == 2
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert all(preds.isna())


@inference_mode
def test_uu_implicit(ml_ratings):
    "Train and use user-user on an implicit data set."
    algo = UserKNNScorer(k=20, feedback="implicit")
    data = ml_ratings.loc[:, ["user_id", "item_id"]]

    algo.train(from_interactions_df(data))
    assert algo.user_means is None

    mat = algo.user_vectors
    norms = np.linalg.norm(mat.todense(), axis=1)
    assert norms.shape == mat.shape[:1]
    assert np.allclose(norms, 1.0)

    preds = algo(query=50, items=ItemList([1, 2, 42]))
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert all(preds[preds.notna()] > 0)


@mark.slow
def test_uu_save_load_implicit(tmp_path, ml_ratings):
    "Save and load user-user on an implicit data set."
    orig = UserKNNScorer(k=20, feedback="implicit")
    data = ml_ratings.loc[:, ["user_id", "item_id"]]

    orig.train(from_interactions_df(data))
    ser = pickle.dumps(orig)

    algo = pickle.loads(ser)

    assert algo.user_means is None
    assert algo.users == orig.users
    assert algo.items == orig.items


@mark.slow
def test_uu_known_preds(ml_ds: Dataset):
    from lenskit import batch

    uknn = UserKNNScorer(k=30, min_sim=1.0e-6)
    pipe = predict_pipeline(uknn, fallback=False)
    _log.info("training %s on ml data", uknn)
    pipe.train(ml_ds)

    dir = Path(__file__).parent
    pred_file = dir / "user-user-preds.csv"
    _log.info("reading known predictions from %s", pred_file)
    known_preds = pd.read_csv(str(pred_file))
    _log.info("generating %d known predictions", len(known_preds))
    known = ItemListCollection.from_df(known_preds, UserIDKey)

    preds = batch.predict(pipe, known, n_jobs=1)
    preds = preds.to_df().drop(columns=["prediction"], errors="ignore")

    merged = pd.merge(known_preds.rename(columns={"prediction": "expected"}), preds)
    assert len(merged) == len(preds)
    merged["error"] = merged.expected - merged.score
    missing = merged.score.isna() & merged.expected.notna()
    if np.any(missing):
        bad = merged[merged.score.isna() & merged.expected.notna()]
        _log.error("%d missing predictions:\n%s", len(bad), bad)
        fail(f"missing predictions for {np.sum(missing)} items")

    err = merged.error
    err = err[err.notna()]
    if np.any(err.abs() > 0.01):
        bad = merged[merged.error.notna() & (merged.error.abs() > 0.01)]
        _log.error("%d erroneous predictions:\n%s", len(bad), bad)
        fail(f"{len(bad)} erroneous predictions")


@mark.slow
@mark.eval
def test_uu_batch_accuracy(ml_100k: pd.DataFrame):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(UserKNNScorer(k=30), ds, predicts_ratings=True)

    summary = results.list_summary()

    assert results.global_metrics()["MAE"] == approx(0.71, abs=0.05)
    assert summary.loc["RMSE", "mean"] == approx(0.91, abs=0.055)


@mark.slow
@mark.eval
def test_uu_implicit_batch_accuracy(ml_100k: pd.DataFrame):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(UserKNNScorer(k=30, feedback="implicit"), ds)

    summary = results.list_summary()

    assert summary.loc["NDCG", "mean"] >= 0.03


@mark.slow
def test_uu_double_ratings(ml_ratings: pd.DataFrame):
    ml_ratings = ml_ratings.astype({"rating": "f8"})
    ds = from_interactions_df(ml_ratings)
    assert ds.interaction_matrix(format="pandas", field="rating")["rating"].dtype == np.float64
    model = UserKNNScorer(k=30, feedback="explicit")
    pipe = topn_pipeline(model)
    pipe.train(ds)

    # assert model.user_vectors_.dtype == torch.float32
    recs = recommend(pipe, 115, 10)
    assert len(recs) == 10
