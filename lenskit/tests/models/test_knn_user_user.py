# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pytest import approx, fail, mark

import lenskit.util.test as lktu
from lenskit.data import Dataset, ItemList, RecQuery, from_interactions_df
from lenskit.data.bulk import dict_to_df, iter_item_lists
from lenskit.knn import UserKNNScorer
from lenskit.metrics import call_metric, quick_measure_model
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401

_log = logging.getLogger(__name__)


def test_uu_train(ml_ratings, ml_ds):
    algo = UserKNNScorer(30)
    algo.train(ml_ds)

    # we have data structures
    assert algo.user_means_ is not None
    assert algo.user_vectors_ is not None
    assert algo.user_ratings_ is not None

    # it should have computed correct means
    u_stats = ml_ds.user_stats()
    mlmeans = pd.Series(algo.user_means_.numpy(), index=algo.users_.ids(), name="mean")
    mlmeans.index.name = "user"
    umeans, mlmeans = u_stats["mean_rating"].align(mlmeans)
    assert mlmeans.values == approx(umeans.values)

    # we should be able to reconstruct rating values
    uir = ml_ratings.set_index(["user", "item"]).rating
    rates = algo.user_ratings_.tocoo()
    ui_rbdf = pd.DataFrame(
        {
            "user": algo.users_.ids(rates.row),
            "item": algo.items_.ids(rates.col),
            "nrating": rates.data,
        }
    ).set_index(["user", "item"])
    ui_rbdf = ui_rbdf.join(mlmeans)
    ui_rbdf["rating"] = ui_rbdf["nrating"] + ui_rbdf["mean"]
    ui_rbdf["orig_rating"] = uir
    assert ui_rbdf.rating.values == approx(ui_rbdf.orig_rating.values)


def test_uu_predict_one(ml_ds):
    algo = UserKNNScorer(30)
    algo.train(ml_ds)

    preds = algo(query=4, items=ItemList([1016]))
    assert len(preds) == 1
    assert preds.ids() == [1016]
    assert preds.scores() == approx([3.62221550680778])


def test_uu_predict_too_few(ml_ds):
    algo = UserKNNScorer(30, min_nbrs=2)
    algo.train(ml_ds)

    preds = algo(query=4, items=ItemList([2091]))
    assert len(preds) == 1
    assert preds.ids() == [2091]
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert all(preds.isna())


def test_uu_predict_too_few_blended(ml_ds):
    algo = UserKNNScorer(30, min_nbrs=2)
    algo.train(ml_ds)

    preds = algo(query=4, items=ItemList([1016, 2091]))
    assert len(preds) == 2
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert np.isnan(preds.loc[2091])
    assert preds.loc[1016] == approx(3.62221550680778)


def test_uu_predict_live_ratings(ml_ratings):
    algo = UserKNNScorer(30, min_nbrs=2)
    no4 = ml_ratings[ml_ratings.user != 4]
    no4 = from_interactions_df(no4, item_col="item")
    algo.train(no4)

    ratings = ItemList.from_df(
        ml_ratings[ml_ratings.user == 4][["item", "rating"]].rename(columns={"item": "item_id"})
    )

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
    orig = UserKNNScorer(30)
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
    mlmeans = pd.Series(algo.user_means_, index=algo.users_, name="mean")
    mlmeans.index.name = "user"
    umeans, mlmeans = umeans.align(mlmeans)
    assert mlmeans.values == approx(umeans.values)

    # we should be able to reconstruct rating values
    uir = ml_ratings.set_index(["user", "item"]).rating
    rates = algo.user_ratings_.tocoo()
    ui_rbdf = pd.DataFrame(
        {
            "user": algo.users_.ids(rates.row),
            "item": algo.items_.ids(rates.col),
            "nrating": rates.data,
        }
    ).set_index(["user", "item"])
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
    algo = UserKNNScorer(30, min_nbrs=2)
    algo.train(ml_ds)

    preds = algo(query=-28018, items=ItemList([1016, 2091]))
    assert len(preds) == 2
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert all(preds.isna())


def test_uu_implicit(ml_ratings):
    "Train and use user-user on an implicit data set."
    algo = UserKNNScorer(20, feedback="implicit")
    data = ml_ratings.loc[:, ["user", "item"]]

    algo.train(from_interactions_df(data, item_col="item"))
    assert algo.user_means_ is None

    mat = algo.user_vectors_
    norms = torch.linalg.vector_norm(mat.to_dense(), dim=1)
    assert norms.shape == mat.shape[:1]
    assert np.allclose(norms.numpy(), 1.0)

    preds = algo(query=50, items=ItemList([1, 2, 42]))
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert all(preds[preds.notna()] > 0)


@mark.slow
def test_uu_save_load_implicit(tmp_path, ml_ratings):
    "Save and load user-user on an implicit data set."
    orig = UserKNNScorer(20, feedback="implicit")
    data = ml_ratings.loc[:, ["user", "item"]]

    orig.train(from_interactions_df(data, item_col="item"))
    ser = pickle.dumps(orig)

    algo = pickle.loads(ser)

    assert algo.user_means_ is None
    assert algo.users_ == orig.users_
    assert algo.items_ == orig.items_


@mark.slow
def test_uu_known_preds(ml_ds: Dataset):
    from lenskit import batch

    uknn = UserKNNScorer(30, min_sim=1.0e-6)
    _log.info("training %s on ml data", uknn)
    uknn.train(ml_ds)

    dir = Path(__file__).parent
    pred_file = dir / "user-user-preds.csv"
    _log.info("reading known predictions from %s", pred_file)
    known_preds = pd.read_csv(str(pred_file))
    _log.info("generating %d known predictions", len(known_preds))

    preds = {
        user: uknn(user, ItemList(kps, prediction=False))
        for (user, kps) in iter_item_lists(known_preds)
    }
    preds = dict_to_df(preds)

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


def __batch_eval(job):
    from lenskit import batch

    algo, train, test = job
    _log.info("running training")
    algo.train(from_interactions_df(train))
    _log.info("testing %d users", test.user.nunique())
    return batch.predict(algo, test)


@mark.slow
@mark.eval
def test_uu_batch_accuracy(ml_100k: pd.DataFrame):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(UserKNNScorer(30), ds, predicts_ratings=True)

    summary = results.list_summary()

    assert results.global_metrics()["MAE"] == approx(0.71, abs=0.05)
    assert summary.loc["RMSE", "mean"] == approx(0.91, abs=0.055)


@mark.slow
@mark.eval
def test_uu_implicit_batch_accuracy(ml_100k: pd.DataFrame):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(UserKNNScorer(30, feedback="implicit"), ds, predicts_ratings=True)

    summary = results.list_summary()

    assert summary.loc["NDCG", "mean"] >= 0.03
