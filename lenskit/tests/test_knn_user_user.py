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

import lenskit.algorithms.knn.user as knn
import lenskit.util.test as lktu
from lenskit.algorithms import Recommender
from lenskit.algorithms.ranking import TopN
from lenskit.data.dataset import Dataset, from_interactions_df
from lenskit.util import clone
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401

_log = logging.getLogger(__name__)


def test_uu_dft_config():
    algo = knn.UserUser(30)
    assert algo.nnbrs == 30
    assert algo.center
    assert algo.aggregate == "weighted-average"
    assert algo.use_ratings


def test_uu_exp_config():
    algo = knn.UserUser(30, feedback="explicit")
    assert algo.nnbrs == 30
    assert algo.center
    assert algo.aggregate == "weighted-average"
    assert algo.use_ratings


def test_uu_imp_config():
    algo = knn.UserUser(30, feedback="implicit")
    assert algo.nnbrs == 30
    assert not algo.center
    assert algo.aggregate == "sum"
    assert not algo.use_ratings


def test_uu_imp_clone():
    algo = knn.UserUser(30, feedback="implicit")
    a2 = clone(algo)

    assert a2.get_params() == algo.get_params()
    assert a2.__dict__ == algo.__dict__


def test_uu_train(ml_ratings, ml_ds):
    algo = knn.UserUser(30)
    ret = algo.fit(ml_ds)
    assert ret is algo

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
    rates = algo.user_ratings_.to_sparse_coo()
    ui_rbdf = pd.DataFrame(
        {
            "user": algo.users_.ids(rates.indices()[0]),
            "item": algo.items_.ids(rates.indices()[1]),
            "nrating": rates.values(),
        }
    ).set_index(["user", "item"])
    ui_rbdf = ui_rbdf.join(mlmeans)
    ui_rbdf["rating"] = ui_rbdf["nrating"] + ui_rbdf["mean"]
    ui_rbdf["orig_rating"] = uir
    assert ui_rbdf.rating.values == approx(ui_rbdf.orig_rating.values)


def test_uu_train_adapt(ml_ds):
    "Test training an adapted user-user (#129)."
    from lenskit.algorithms import Recommender

    uu = knn.UserUser(30)
    uu = Recommender.adapt(uu)
    ret = uu.fit(ml_ds)
    assert isinstance(uu, TopN)
    assert ret is uu
    assert isinstance(uu.predictor, knn.UserUser)


def test_uu_predict_one(ml_ds):
    algo = knn.UserUser(30)
    algo.fit(ml_ds)

    preds = algo.predict_for_user(4, [1016])
    assert len(preds) == 1
    assert preds.index == [1016]
    assert preds.values == approx([3.62221550680778])


def test_uu_predict_too_few(ml_ds):
    algo = knn.UserUser(30, min_nbrs=2)
    algo.fit(ml_ds)

    preds = algo.predict_for_user(4, [2091])
    assert len(preds) == 1
    assert preds.index == [2091]
    assert all(preds.isna())


def test_uu_predict_too_few_blended(ml_ds):
    algo = knn.UserUser(30, min_nbrs=2)
    algo.fit(ml_ds)

    preds = algo.predict_for_user(4, [1016, 2091])
    assert len(preds) == 2
    assert np.isnan(preds.loc[2091])
    assert preds.loc[1016] == approx(3.62221550680778)


def test_uu_predict_live_ratings(ml_ratings):
    algo = knn.UserUser(30, min_nbrs=2)
    no4 = ml_ratings[ml_ratings.user != 4]
    algo.fit(from_interactions_df(no4, item_col="item"))

    ratings = ml_ratings[ml_ratings.user == 4].set_index("item").rating

    preds = algo.predict_for_user(20381, [1016, 2091], ratings)
    assert len(preds) == 2
    assert np.isnan(preds.loc[2091])
    assert preds.loc[1016] == approx(3.62221550680778)


def test_uu_save_load(tmp_path, ml_ratings, ml_ds):
    orig = knn.UserUser(30)
    _log.info("training model")
    orig.fit(ml_ds)

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
    rates = algo.user_ratings_.to_sparse_coo()
    ui_rbdf = pd.DataFrame(
        {
            "user": algo.users_.ids(rates.indices()[0]),
            "item": algo.items_.ids(rates.indices()[1]),
            "nrating": rates.values(),
        }
    ).set_index(["user", "item"])
    ui_rbdf = ui_rbdf.join(mlmeans)
    ui_rbdf["rating"] = ui_rbdf["nrating"] + ui_rbdf["mean"]
    ui_rbdf["orig_rating"] = uir
    assert ui_rbdf.rating.values == approx(ui_rbdf.orig_rating.values)

    # running the predictor should work
    preds = algo.predict_for_user(4, [1016])
    assert len(preds) == 1
    assert preds.index == [1016]
    assert preds.values == approx([3.62221550680778])


def test_uu_predict_unknown_empty(ml_ds):
    algo = knn.UserUser(30, min_nbrs=2)
    algo.fit(ml_ds)

    preds = algo.predict_for_user(-28018, [1016, 2091])
    assert len(preds) == 2
    assert all(preds.isna())


def test_uu_implicit(ml_ratings):
    "Train and use user-user on an implicit data set."
    algo = knn.UserUser(20, feedback="implicit")
    data = ml_ratings.loc[:, ["user", "item"]]

    algo.fit(from_interactions_df(data, item_col="item"))
    assert algo.user_means_ is None

    mat = algo.user_vectors_
    norms = torch.linalg.vector_norm(mat.to_dense(), dim=1)
    assert norms.shape == mat.shape[:1]
    assert np.allclose(norms.numpy(), 1.0)

    preds = algo.predict_for_user(50, [1, 2, 42])
    assert all(preds[preds.notna()] > 0)


@mark.slow
def test_uu_save_load_implicit(tmp_path, ml_ratings):
    "Save and load user-user on an implicit data set."
    orig = knn.UserUser(20, feedback="implicit")
    data = ml_ratings.loc[:, ["user", "item"]]

    orig.fit(from_interactions_df(data, item_col="item"))
    ser = pickle.dumps(orig)

    algo = pickle.loads(ser)

    assert algo.user_means_ is None
    assert algo.users_ == orig.users_
    assert algo.items_ == orig.items_


@mark.slow
def test_uu_known_preds(ml_ds: Dataset):
    from lenskit import batch

    algo = knn.UserUser(30, min_sim=1.0e-6)
    _log.info("training %s on ml data", algo)
    algo.fit(ml_ds)

    dir = Path(__file__).parent
    pred_file = dir / "user-user-preds.csv"
    _log.info("reading known predictions from %s", pred_file)
    known_preds = pd.read_csv(str(pred_file))
    pairs = known_preds.loc[:, ["user", "item"]]
    _log.info("generating %d known predictions", len(pairs))

    preds = batch.predict(algo, pairs, n_jobs=1)
    merged = pd.merge(known_preds.rename(columns={"prediction": "expected"}), preds)
    assert len(merged) == len(preds)
    merged["error"] = merged.expected - merged.prediction
    missing = merged.prediction.isna() & merged.expected.notna()
    if np.any(missing):
        bad = merged[merged.prediction.isna() & merged.expected.notna()]
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
    algo.fit(from_interactions_df(train))
    _log.info("testing %d users", test.user.nunique())
    return batch.predict(algo, test)


@mark.slow
@mark.eval
def test_uu_batch_accuracy(ml_100k: pd.DataFrame):
    import lenskit.crossfold as xf
    import lenskit.metrics.predict as pm
    from lenskit.algorithms import basic, bias

    uu_algo = knn.UserUser(30)
    algo = basic.Fallback(uu_algo, bias.Bias())

    folds = xf.partition_users(ml_100k, 5, xf.SampleFrac(0.2))
    preds = [__batch_eval((algo, train, test)) for (train, test) in folds]
    preds = pd.concat(preds)
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.71, abs=0.05)

    user_rmse = preds.groupby("user").apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.91, abs=0.055)


@mark.slow
@mark.eval
def test_uu_implicit_batch_accuracy(ml_100k: pd.DataFrame):
    import lenskit.crossfold as xf
    from lenskit import batch, topn

    algo = knn.UserUser(30, center=False, aggregate="sum")

    folds = list(xf.partition_users(ml_100k, 5, xf.SampleFrac(0.2)))
    all_test = pd.concat(f.test for f in folds)

    rec_lists = []
    for train, test in folds:
        _log.info("running training")
        rec_algo = Recommender.adapt(algo)
        rec_algo.fit(from_interactions_df(train.loc[:, ["user", "item"]]))
        _log.info("testing %d users", test.user.nunique())
        recs = batch.recommend(rec_algo, test.user.unique(), 100, n_jobs=2)
        rec_lists.append(recs)
    recs = pd.concat(rec_lists)

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, all_test)
    user_dcg = results.ndcg

    dcg = user_dcg.mean()
    assert dcg >= 0.03
