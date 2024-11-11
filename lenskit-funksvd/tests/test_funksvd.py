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

from pytest import approx, mark

import lenskit.funksvd as svd
import lenskit.util.test as lktu
from lenskit.data import Dataset, from_interactions_df
from lenskit.metrics import call_metric

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


def test_fsvd_basic_build():
    algo = svd.FunkSVD(20, iterations=20)
    algo.fit(simple_ds)

    assert algo.bias is not None
    assert algo.bias.mean_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)


def test_fsvd_clamp_build():
    algo = svd.FunkSVD(20, iterations=20, range=(1, 5))
    algo.fit(simple_ds)

    assert algo.bias is not None
    assert algo.bias.mean_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)


def test_fsvd_predict_basic():
    algo = svd.FunkSVD(20, iterations=20)
    algo.fit(simple_ds)

    assert algo.bias is not None
    assert algo.bias.mean_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3] <= 5


def test_fsvd_predict_clamp():
    algo = svd.FunkSVD(20, iterations=20, range=(1, 5))
    algo.fit(simple_ds)

    assert algo.bias is not None
    assert algo.bias.mean_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [3])
    assert isinstance(preds, pd.Series)
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= 1
    assert preds.loc[3] <= 5


def test_fsvd_no_bias():
    algo = svd.FunkSVD(20, iterations=20, bias=None)
    algo.fit(simple_ds)

    assert algo.bias is None
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert all(preds.notna())


def test_fsvd_predict_bad_item():
    algo = svd.FunkSVD(20, iterations=20)
    algo.fit(simple_ds)

    assert algo.bias is not None
    assert algo.bias.mean_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_fsvd_predict_bad_item_clamp():
    algo = svd.FunkSVD(20, iterations=20, range=(1, 5))
    algo.fit(simple_ds)

    assert algo.bias is not None
    assert algo.bias.mean_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_fsvd_predict_bad_user():
    algo = svd.FunkSVD(20, iterations=20)
    algo.fit(simple_ds)

    assert algo.bias is not None
    assert algo.bias.mean_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@lktu.wantjit
@mark.slow
def test_fsvd_save_load(ml_ds: Dataset):
    original = svd.FunkSVD(20, iterations=20)
    original.fit(ml_ds)

    assert original.bias is not None
    assert original.bias.mean_ == approx(
        ml_ds.interaction_matrix("scipy", field="rating").data.mean()
    )
    assert original.item_features_.shape == (ml_ds.item_count, 20)
    assert original.user_features_.shape == (ml_ds.user_count, 20)

    mod = pickle.dumps(original)
    _log.info("serialized to %d bytes", len(mod))
    algo = pickle.loads(mod)

    assert algo.bias.mean_ == original.bias.mean_
    assert np.all(algo.bias.user_offsets_ == original.bias.user_offsets_)
    assert np.all(algo.bias.item_offsets_ == original.bias.item_offsets_)
    assert np.all(algo.user_features_ == original.user_features_)
    assert np.all(algo.item_features_ == original.item_features_)
    assert np.all(algo.items_.index == original.items_.index)
    assert np.all(algo.users_.index == original.users_.index)


@lktu.wantjit
@mark.slow
def test_fsvd_train_binary(ml_ratings: pd.DataFrame):
    ratings = ml_ratings.drop(columns=["rating", "timestamp"])

    original = svd.FunkSVD(20, iterations=20, bias=False)
    original.fit(from_interactions_df(ratings))

    assert original.bias is None
    assert original.item_features_.shape == (ratings.item.nunique(), 20)
    assert original.user_features_.shape == (ratings.user.nunique(), 20)


@lktu.wantjit
@mark.slow
def test_fsvd_known_preds(ml_ds: Dataset):
    algo = svd.FunkSVD(15, iterations=125, lrate=0.001)
    _log.info("training %s on ml data", algo)
    algo.fit(ml_ds)

    dir = Path(__file__).parent
    pred_file = dir / "funksvd-preds.csv"
    _log.info("reading known predictions from %s", pred_file)
    known_preds = pd.read_csv(str(pred_file))
    pairs = known_preds.loc[:, ["user", "item"]]

    preds = algo.predict(pairs)
    known_preds.rename(columns={"prediction": "expected"}, inplace=True)
    merged = known_preds.assign(prediction=preds)
    merged["error"] = merged.expected - merged.prediction
    assert not any(merged.prediction.isna() & merged.expected.notna())
    err = merged.error
    err = err[err.notna()]
    try:
        assert all(err.abs() < 0.01)
    except AssertionError as e:
        bad = merged[merged.error.notna() & (merged.error.abs() >= 0.01)]
        _log.error("erroneous predictions:\n%s", bad)
        raise e


@lktu.wantjit
@mark.slow
@mark.eval
def test_fsvd_batch_accuracy(ml_100k: pd.DataFrame):
    import lenskit.crossfold as xf
    import lenskit.metrics.predict as pm
    from lenskit import batch
    from lenskit.algorithms import basic, bias

    svd_algo = svd.FunkSVD(25, 125, damping=10)
    algo = basic.Fallback(svd_algo, bias.Bias(damping=10))

    def eval(train, test):
        _log.info("running training")
        algo.fit(from_interactions_df(train))
        _log.info("testing %d users", test.user.nunique())
        return batch.predict(algo, test)

    folds = xf.partition_users(ml_100k, 5, xf.SampleFrac(0.2))
    preds = pd.concat(eval(train, test) for (train, test) in folds)
    mae = call_metric(pm.MAE, preds)
    assert mae == approx(0.74, abs=0.025)

    user_rmse = pm.measure_user_predictions(preds, pm.RMSE)
    assert user_rmse.mean() == approx(0.92, abs=0.05)
