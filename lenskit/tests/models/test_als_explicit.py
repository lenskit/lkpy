# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd
import torch

from pytest import approx, mark

from lenskit.als import BiasedMFScorer
from lenskit.data import Dataset, ItemList, RecQuery, from_interactions_df, load_movielens_df
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests, wantjit

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


class TestExplicitALS(BasicComponentTests, ScorerTests):
    component = BiasedMFScorer


def test_als_basic_build():
    algo = BiasedMFScorer(features=20, epochs=10)
    algo.train(simple_ds)

    assert algo.bias_ is not None
    assert algo.users_ is not None
    assert algo.user_features_ is not None

    assert algo.bias_.global_bias == approx(simple_df.rating.mean())
    assert set(algo.users_.ids()) == set([10, 12, 13])
    assert set(algo.items_.ids()) == set([1, 2, 3])
    assert algo.user_features_.shape == (3, 20)
    assert algo.item_features_.shape == (3, 20)

    assert algo.config.features == 20
    assert len(algo.users_) == 3
    assert len(algo.items_) == 3


def test_als_predict_basic():
    algo = BiasedMFScorer(features=20, epochs=10)
    algo.train(simple_ds)

    assert algo.bias_ is not None

    assert algo.bias_.global_bias == approx(simple_df.rating.mean())

    preds = algo(query=10, items=ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5.1


def test_als_predict_basic_for_new_ratings():
    algo = BiasedMFScorer(features=20, epochs=10)
    algo.train(simple_ds)

    assert algo.bias_ is not None
    assert algo.bias_.global_bias == approx(simple_df.rating.mean())

    query = RecQuery(15, ItemList(item_ids=[1, 2], rating=[4.0, 5.0]))
    preds = algo(query, items=ItemList([3]))

    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5.1


def test_als_predict_basic_for_new_user_with_new_ratings():
    u = 10
    i = 3

    algo = BiasedMFScorer(features=20, epochs=10)
    algo.train(simple_ds)

    preds = algo(query=u, items=ItemList([i]))
    preds = preds.scores("pandas", index="ids")
    assert preds is not None

    query = RecQuery(-1, ItemList(item_ids=[1, 2], rating=[4.0, 5.0]))

    new_preds = algo(query=query, items=ItemList([i]))
    new_preds = new_preds.scores("pandas", index="ids")
    assert new_preds is not None

    assert preds.loc[i] == approx(new_preds.loc[i], rel=9e-2)


def test_als_predict_for_new_users_with_new_ratings(rng, ml_ds: Dataset):
    n_users = 3
    n_items = 2

    users = rng.choice(ml_ds.users.ids(), n_users)
    items = rng.choice(ml_ds.items.ids(), n_items)

    algo = BiasedMFScorer(features=20, epochs=10)
    algo.train(ml_ds)

    _log.debug("Items: " + str(items))
    assert algo.bias_ is not None
    assert algo.users_ is not None
    assert algo.user_features_ is not None

    for u in users:
        _log.debug(f"user: {u}")
        preds = algo(query=u, items=ItemList(items))
        assert np.all(preds.ids() == items)

        user_data = ml_ds.user_row(u)

        _log.debug("user_features from fit: " + str(algo.user_features_[algo.users_.number(u), :]))

        query = RecQuery(-1, user_data)
        new_preds = algo(query=query, items=ItemList(items))
        assert np.all(new_preds.ids() == items)

        _log.debug("preds: " + str(preds.scores()))
        _log.debug("new_preds: " + str(new_preds.scores()))
        _log.debug("------------")
        assert new_preds.scores() == approx(preds.scores(), rel=9e-2)


def test_als_predict_bad_item():
    algo = BiasedMFScorer(features=20, epochs=10)
    algo.train(simple_ds)

    assert algo.bias_ is not None
    assert algo.bias_.global_bias == approx(simple_df.rating.mean())

    preds = algo(query=10, items=ItemList([4]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_als_predict_bad_user():
    algo = BiasedMFScorer(features=20, epochs=10)
    algo.train(simple_ds)

    assert algo.bias_ is not None
    assert algo.bias_.global_bias == approx(simple_df.rating.mean())

    preds = algo(query=50, items=ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


def test_als_predict_no_user_features_basic(rng: np.random.Generator, ml_ds: Dataset):
    n_items = 2

    u = rng.choice(ml_ds.users.ids(), 1).item()
    items = rng.choice(ml_ds.items.ids(), n_items)

    algo = BiasedMFScorer(features=5, epochs=10)
    algo.train(ml_ds)
    _log.debug("Items: " + str(items))
    assert algo.bias_ is not None
    assert algo.users_ is not None
    assert algo.user_features_ is not None

    algo_no_user_features = BiasedMFScorer(features=5, epochs=10, save_user_features=False)
    algo_no_user_features.train(ml_ds)

    assert algo_no_user_features.user_features_ is None

    _log.debug(f"user: {u}")
    preds = algo(query=u, items=ItemList(item_ids=items))

    user_data = ml_ds.user_row(u)

    _log.debug("user_features from fit: " + str(algo.user_features_[algo.users_.number(u), :]))

    query = RecQuery(-1, user_data)
    new_preds = algo_no_user_features(query, items=ItemList(items))

    _log.debug("preds: " + str(preds.scores()))
    _log.debug("new_preds: " + str(new_preds.scores()))
    _log.debug("------------")
    assert new_preds.scores() == approx(preds.scores(), rel=9e-1)


@wantjit
@mark.slow
def test_als_train_large(ml_ratings, ml_ds: Dataset):
    algo = BiasedMFScorer(features=20, epochs=10)
    algo.train(ml_ds)

    assert algo.bias_ is not None
    assert algo.users_ is not None
    assert algo.user_features_ is not None

    assert algo.bias_.global_bias == approx(ml_ratings.rating.mean())
    assert algo.config.features == 20
    assert len(algo.items_) == ml_ds.item_count
    assert len(algo.users_) == ml_ds.user_count

    ratings = ml_ds.interaction_matrix(format="pandas")
    gmean = ratings["rating"].mean()

    istats = ml_ds.item_stats()
    icounts = istats["rating_count"]
    isums = istats["mean_rating"] * icounts
    is2 = isums - icounts * gmean
    imeans = is2 / (icounts + 5)
    ibias = pd.Series(algo.bias_.item_biases, index=algo.items_.index)
    imeans, ibias = imeans.align(ibias)
    assert ibias.values == approx(imeans.values, rel=1.0e-3)


# don't use wantjit, use this to do a non-JIT test
def test_als_save_load(ml_ds: Dataset):
    original = BiasedMFScorer(features=5, epochs=5)
    original.train(ml_ds)

    assert original.bias_ is not None
    assert original.users_ is not None

    mod = pickle.dumps(original)
    _log.info("serialized to %d bytes", len(mod))

    algo = pickle.loads(mod)
    assert algo.bias_.global_bias == original.bias_.global_bias
    assert np.all(algo.bias_.user_biases == original.bias_.user_biases)
    assert np.all(algo.bias_.item_biases == original.bias_.item_biases)
    assert torch.all(algo.user_features_ == original.user_features_)
    assert torch.all(algo.item_features_ == original.item_features_)
    assert np.all(algo.items_.index == original.items_.index)
    assert np.all(algo.users_.index == original.users_.index)

    # make sure it still works
    preds = algo(query=10, items=ItemList(np.arange(0, 50, dtype="i8")))
    assert len(preds) == 50


@mark.slow
@mark.eval
def test_als_batch_accuracy(ml_100k):
    ds = from_interactions_df(ml_100k)
    results = quick_measure_model(
        BiasedMFScorer(features=25, epochs=20, damping=5), ds, predicts_ratings=True
    )

    assert results.global_metrics()["MAE"] == approx(0.73, abs=0.045)
    assert results.list_summary().loc["RMSE", "mean"] == approx(0.94, abs=0.05)
