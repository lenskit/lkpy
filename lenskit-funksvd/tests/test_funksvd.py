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

from lenskit.data import Dataset, ItemList, from_interactions_df
from lenskit.data.bulk import dict_to_df, iter_item_lists
from lenskit.funksvd import FunkSVD
from lenskit.metrics import call_metric, quick_measure_model
from lenskit.util.test import ml_100k, ml_ds, wantjit  # noqa: F401

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


def test_fsvd_basic_build():
    algo = FunkSVD(20, iterations=20)
    algo.train(simple_ds)

    assert algo.bias_ is not None
    assert algo.bias_.global_bias == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)


def test_fsvd_clamp_build():
    algo = FunkSVD(20, iterations=20, range=(1, 5))
    algo.train(simple_ds)

    assert algo.bias_ is not None
    assert algo.bias_.global_bias == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)


def test_fsvd_predict_basic():
    algo = FunkSVD(20, iterations=20)
    algo.train(simple_ds)

    assert algo.bias_ is not None
    assert algo.bias_.global_bias == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo(query=10, items=ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3] <= 5


def test_fsvd_predict_clamp():
    algo = FunkSVD(20, iterations=20, range=(1, 5))
    algo.train(simple_ds)

    assert algo.bias_ is not None
    assert algo.bias_.global_bias == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo(query=10, items=ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert preds.loc[3] >= 1
    assert preds.loc[3] <= 5


def test_fsvd_predict_bad_item():
    algo = FunkSVD(20, iterations=20)
    algo.train(simple_ds)

    assert algo.bias_ is not None
    assert algo.bias_.global_bias == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo(10, ItemList([4]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_fsvd_predict_bad_item_clamp():
    algo = FunkSVD(20, iterations=20, range=(1, 5))
    algo.train(simple_ds)

    assert algo.bias_ is not None
    assert algo.bias_.global_bias == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo(10, ItemList([4]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_fsvd_predict_bad_user():
    algo = FunkSVD(20, iterations=20)
    algo.train(simple_ds)

    assert algo.bias_ is not None
    assert algo.bias_.global_bias == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo(query=50, items=ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@wantjit
@mark.slow
def test_fsvd_save_load(ml_ds: Dataset):
    original = FunkSVD(20, iterations=20)
    original.train(ml_ds)

    assert original.bias_ is not None
    assert original.bias_.global_bias == approx(
        ml_ds.interaction_matrix("scipy", field="rating").data.mean()
    )
    assert original.item_features_.shape == (ml_ds.item_count, 20)
    assert original.user_features_.shape == (ml_ds.user_count, 20)

    mod = pickle.dumps(original)
    _log.info("serialized to %d bytes", len(mod))
    algo = pickle.loads(mod)

    assert algo.bias_.global_bias == original.bias_.global_bias
    assert np.all(algo.bias_.user_biases == original.bias_.user_biases)
    assert np.all(algo.bias_.item_biases == original.bias_.item_biases)
    assert np.all(algo.user_features_ == original.user_features_)
    assert np.all(algo.item_features_ == original.item_features_)
    assert np.all(algo.items_.index == original.items_.index)
    assert np.all(algo.users_.index == original.users_.index)


@wantjit
@mark.slow
def test_fsvd_known_preds(ml_ds: Dataset):
    algo = FunkSVD(15, iterations=125, lrate=0.001)
    _log.info("training %s on ml data", algo)
    algo.train(ml_ds)

    dir = Path(__file__).parent
    pred_file = dir / "funksvd-preds.csv"
    _log.info("reading known predictions from %s", pred_file)
    known_preds = pd.read_csv(str(pred_file))

    preds = {u: algo(u, il) for (u, il) in iter_item_lists(known_preds)}
    preds = dict_to_df(preds)

    known_preds.rename(columns={"prediction": "expected"}, inplace=True)
    merged = pd.merge(known_preds, preds)

    merged["error"] = merged.expected - merged.score
    assert not any(merged.score.isna() & merged.expected.notna())
    err = merged.error
    err = err[err.notna()]
    try:
        assert all(err.abs() < 0.01)
    except AssertionError as e:
        bad = merged[merged.error.notna() & (merged.error.abs() >= 0.01)]
        _log.error("erroneous predictions:\n%s", bad)
        raise e


@wantjit
@mark.slow
@mark.eval
def test_fsvd_batch_accuracy(ml_100k: pd.DataFrame):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(FunkSVD(25, 125, damping=10), ds, predicts_ratings=True)

    assert results.global_metrics().loc["MAE"] == approx(0.74, abs=0.025)
    assert results.list_summary().loc["RMSE", "mean"] == approx(0.92, abs=0.05)
