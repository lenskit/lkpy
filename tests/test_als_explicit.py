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

import lenskit.util.test as lktu
from lenskit.algorithms import als

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)


def test_als_basic_build():
    algo = als.BiasedMF(20, epochs=10)
    algo.fit(simple_df)

    assert algo.bias.mean_ == approx(simple_df.rating.mean())
    assert set(algo.user_index_) == set([10, 12, 13])
    assert set(algo.item_index_) == set([1, 2, 3])
    assert algo.user_features_.shape == (3, 20)
    assert algo.item_features_.shape == (3, 20)

    assert algo.n_features == 20
    assert algo.n_users == 3
    assert algo.n_items == 3


def test_als_no_bias():
    algo = als.BiasedMF(20, epochs=10, bias=None)
    algo.fit(simple_df)

    assert algo.bias is None
    assert set(algo.user_index_) == set([10, 12, 13])
    assert set(algo.item_index_) == set([1, 2, 3])
    assert algo.user_features_.shape == (3, 20)
    assert algo.item_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1


def test_als_predict_basic():
    algo = als.BiasedMF(20, epochs=10)
    algo.fit(simple_df)

    assert algo.bias.mean_ == approx(simple_df.rating.mean())

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5.1


def test_als_predict_basic_for_new_ratings():
    algo = als.BiasedMF(20, epochs=10)
    algo.fit(simple_df)

    assert algo.bias.mean_ == approx(simple_df.rating.mean())

    new_ratings = pd.Series([4.0, 5.0], index=[1, 2])  # items as index and ratings as values

    preds = algo.predict_for_user(15, [3], new_ratings)

    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5.1


def test_als_predict_basic_for_new_user_with_new_ratings():
    u = 10
    i = 3

    algo = als.BiasedMF(20, epochs=10)
    algo.fit(simple_df)

    preds = algo.predict_for_user(u, [i])

    new_u_id = -1
    new_ratings = pd.Series([4.0, 5.0], index=[1, 2])  # items as index and ratings as values

    new_preds = algo.predict_for_user(new_u_id, [i], new_ratings)

    assert preds.loc[i] == approx(new_preds.loc[i], rel=9e-2)


def test_als_predict_for_new_users_with_new_ratings():
    n_users = 3
    n_items = 2
    new_u_id = -1
    ratings = lktu.ml_test.ratings

    np.random.seed(45)
    users = np.random.choice(ratings.user.unique(), n_users)
    items = np.random.choice(ratings.item.unique(), n_items)

    algo = als.BiasedMF(20, epochs=10)
    algo.fit(ratings)
    _log.debug("Items: " + str(items))

    for u in users:
        _log.debug(f"user: {u}")
        preds = algo.predict_for_user(u, items)

        user_data = ratings[ratings.user == u]

        _log.debug(
            "user_features from fit: " + str(algo.user_features_[algo.user_index_.get_loc(u), :])
        )

        new_ratings = pd.Series(
            user_data.rating.to_numpy(), index=user_data.item
        )  # items as index and ratings as values
        new_preds = algo.predict_for_user(new_u_id, items, new_ratings)

        _log.debug("preds: " + str(preds.values))
        _log.debug("new_preds: " + str(new_preds.values))
        _log.debug("------------")
        assert new_preds.values == approx(preds.values, rel=9e-2)


def test_als_predict_bad_item():
    algo = als.BiasedMF(20, epochs=10)
    algo.fit(simple_df)

    assert algo.bias.mean_ == approx(simple_df.rating.mean())

    preds = algo.predict_for_user(10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_als_predict_bad_user():
    algo = als.BiasedMF(20, epochs=10)
    algo.fit(simple_df)

    assert algo.bias.mean_ == approx(simple_df.rating.mean())

    preds = algo.predict_for_user(50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


def test_als_predict_no_user_features_basic():
    n_users = 1
    n_items = 2
    new_u_id = -1
    ratings = lktu.ml_test.ratings

    np.random.seed(45)
    u = np.random.choice(ratings.user.unique(), n_users)[0]
    items = np.random.choice(ratings.item.unique(), n_items)

    algo = als.BiasedMF(5, epochs=10)
    algo.fit(ratings)
    _log.debug("Items: " + str(items))

    algo_no_user_features = als.BiasedMF(5, epochs=10, save_user_features=False)
    algo_no_user_features.fit(ratings)

    assert algo_no_user_features.user_features_ is None

    _log.debug(f"user: {u}")
    preds = algo.predict_for_user(u, items)

    user_data = ratings[ratings.user == u]

    _log.debug(
        "user_features from fit: " + str(algo.user_features_[algo.user_index_.get_loc(u), :])
    )

    new_ratings = pd.Series(
        user_data.rating.to_numpy(), index=user_data.item
    )  # items as index and ratings as values
    new_preds = algo_no_user_features.predict_for_user(new_u_id, items, new_ratings)

    _log.debug("preds: " + str(preds.values))
    _log.debug("new_preds: " + str(new_preds.values))
    _log.debug("------------")
    assert new_preds.values == approx(preds.values, rel=9e-1)


@lktu.wantjit
@mark.slow
def test_als_train_large():
    algo = als.BiasedMF(20, epochs=10)
    ratings = lktu.ml_test.ratings
    algo.fit(ratings)

    assert algo.bias.mean_ == approx(ratings.rating.mean())
    assert algo.n_features == 20
    assert algo.n_items == ratings.item.nunique()
    assert algo.n_users == ratings.user.nunique()

    icounts = ratings.groupby("item").rating.count()
    isums = ratings.groupby("item").rating.sum()
    is2 = isums - icounts * ratings.rating.mean()
    imeans = is2 / (icounts + 5)
    ibias = pd.Series(algo.bias.item_offsets_, index=algo.item_index_)
    imeans, ibias = imeans.align(ibias)
    assert ibias.values == approx(imeans.values)


# don't use wantjit, use this to do a non-JIT test
def test_als_save_load():
    original = als.BiasedMF(5, epochs=5)
    ratings = lktu.ml_test.ratings
    original.fit(ratings)

    assert original.bias.mean_ == approx(ratings.rating.mean())

    mod = pickle.dumps(original)
    _log.info("serialized to %d bytes", len(mod))

    algo = pickle.loads(mod)
    assert algo.bias.mean_ == original.bias.mean_
    assert np.all(algo.bias.user_offsets_ == original.bias.user_offsets_)
    assert np.all(algo.bias.item_offsets_ == original.bias.item_offsets_)
    assert torch.all(algo.user_features_ == original.user_features_)
    assert torch.all(algo.item_features_ == original.item_features_)
    assert np.all(algo.item_index_ == original.item_index_)
    assert np.all(algo.user_index_ == original.user_index_)

    # make sure it still works
    preds = algo.predict_for_user(10, np.arange(0, 50, dtype="i8"))
    assert len(preds) == 50


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason="ML100K data not present")
def test_als_batch_accuracy():
    import lenskit.crossfold as xf
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.ratings

    lu_algo = als.BiasedMF(25, epochs=20, damping=5)
    # algo = bias.Fallback(svd_algo, bias.Bias(damping=5))

    def eval(train, test):
        _log.info("training LU")
        lu_algo.fit(train)
        _log.info("testing %d users", test.user.nunique())
        return test.assign(lu_pred=lu_algo.predict(test))

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    preds = pd.concat(eval(train, test) for (train, test) in folds)
    # preds["abs_diff"] = np.abs(preds.lu_pred - preds.cd_pred)
    # _log.info("predictions:\n%s", preds.sort_values("abs_diff", ascending=False))
    # _log.info("diff summary:\n%s", preds.abs_diff.describe())

    lu_mae = pm.mae(preds.lu_pred, preds.rating)
    assert lu_mae == approx(0.73, abs=0.045)

    user_rmse = preds.groupby("user").apply(lambda df: pm.rmse(df.lu_pred, df.rating))
    assert user_rmse.mean() == approx(0.94, abs=0.05)
