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
from lenskit.algorithms import Recommender, als
from lenskit.data import Dataset, from_interactions_df, load_movielens_df

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({"item": [1, 1, 2, 3], "user": [10, 12, 10, 13]})
simple_ds = from_interactions_df(simple_df)

simple_dfr = simple_df.assign(rating=[4.0, 3.0, 5.0, 2.0])
simple_dsr = from_interactions_df(simple_dfr)


def test_als_basic_build():
    algo = als.ImplicitMF(20, epochs=10)
    algo.fit(simple_ds)

    assert algo.users_ is not None
    assert algo.user_features_ is not None

    assert set(algo.users_.ids()) == set([10, 12, 13])
    assert set(algo.items_.ids()) == set([1, 2, 3])
    assert algo.user_features_.shape == (3, 20)
    assert algo.item_features_.shape == (3, 20)


def test_als_predict_basic():
    algo = als.ImplicitMF(20, epochs=10)
    algo.fit(simple_ds)

    preds = algo.predict_for_user(10, [3])

    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5


def test_als_predict_basic_for_new_ratings():
    """Test ImplicitMF ability to support new ratings"""
    algo = als.ImplicitMF(20, epochs=10)
    algo.fit(simple_ds)

    new_ratings = pd.Series([4.0, 5.0], index=[1, 2])  # items as index and ratings as values

    preds = algo.predict_for_user(15, [3], new_ratings)

    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5


def test_als_predict_basic_for_new_user_with_new_ratings():
    """
    Test if ImplicitMF predictions using the same ratings for a new user
    is the same as a user in the current simple_df dataset.
    """
    u = 10
    i = 3

    algo = als.ImplicitMF(20, epochs=10, use_ratings=True)
    algo.fit(simple_dsr)

    preds = algo.predict_for_user(u, [i])

    new_u_id = 1
    new_ratings = pd.Series([4.0, 5.0], index=[1, 2])  # items as index and ratings as values

    new_preds = algo.predict_for_user(new_u_id, [i], new_ratings)
    assert abs(preds.loc[i] - new_preds.loc[i]) <= 0.1


def test_als_predict_for_new_users_with_new_ratings(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    """
    Test if ImplicitMF predictions using the same ratings for a new user
    is the same as a user in ml-latest-small dataset.
    The test is run for more than one user.
    """
    n_users = 3
    n_items = 2
    new_u_id = -1

    np.random.seed(45)
    users = np.random.choice(ml_ds.users, n_users)
    items = np.random.choice(ml_ds.items, n_items)

    algo = als.ImplicitMF(20, epochs=10, use_ratings=False)
    algo.fit(ml_ds)
    assert algo.users_ is not None
    assert algo.user_features_ is not None

    _log.debug("Items: " + str(items))

    for u in users:
        _log.debug(f"user: {u}")
        preds = algo.predict_for_user(u, items)
        upos = algo.users_.number(u)

        # get the user's rating series
        user_data = ml_ratings[ml_ratings.user == u]
        new_ratings = user_data.set_index("item")["rating"].copy()

        nr_info = new_ratings.to_frame()
        ifs = algo.item_features_[algo.items_.index.get_indexer_for(nr_info.index), :]
        fit_uv = algo.user_features_[upos, :]
        nr_info["fit_recon"] = ifs @ fit_uv
        nr_info["fit_sqerr"] = np.square(algo.weight + 1.0 - nr_info["fit_recon"])

        _log.debug("user_features from fit:\n%s", fit_uv)
        new_uv, _new_off = algo.new_user_embedding(new_u_id, new_ratings)
        nr_info["new_recon"] = ifs @ new_uv
        nr_info["new_sqerr"] = np.square(algo.weight + 1.0 - nr_info["new_recon"])

        _log.debug("user features from new:\n%s", new_uv)

        _log.debug("training data reconstruction:\n%s", nr_info)

        new_preds = algo.predict_for_user(new_u_id, items, new_ratings)

        _log.debug("preds: " + str(preds.values))
        _log.debug("new_preds: " + str(new_preds.values))
        _log.debug("------------")

        diffs = np.abs(preds - new_preds)
        assert all(diffs <= 0.1)


def test_als_recs_topn_for_new_users_with_new_ratings(
    rng, ml_ratings: pd.DataFrame, ml_ds: Dataset
):
    """
    Test if ImplicitMF topn recommendations using the same ratings for a new user
    is the same as a user in ml-latest-small dataset.
    The test is run for more than one user.
    """
    import scipy.stats as stats

    from lenskit.algorithms import basic

    n_users = 10
    new_u_id = -1

    users = rng.choice(ml_ds.users, n_users)

    algo = als.ImplicitMF(20, epochs=10, use_ratings=True)
    rec_algo = basic.TopN(algo)
    rec_algo.fit(ml_ds)
    assert algo.users_ is not None
    assert algo.user_features_ is not None
    # _log.debug("Items: " + str(items))

    correlations = pd.Series(np.nan, index=users)
    for u in users:
        recs = rec_algo.recommend(u, 10)
        user_data = ml_ratings[ml_ratings.user == u]
        upos = algo.users_.number(u)
        _log.info("user %s: %s ratings", u, len(user_data))

        _log.debug("user_features from fit: " + str(algo.user_features_[upos, :]))

        # get the user's rating series
        new_ratings = user_data.set_index("item")["rating"].copy()
        new_recs = rec_algo.recommend(new_u_id, 10, ratings=new_ratings)

        # merge new & old recs
        all_recs = pd.merge(
            recs.rename(columns={"score": "old_score"}),
            new_recs.rename(columns={"score": "new_score"}),
            how="outer",
        ).fillna(-np.inf)

        tau = stats.kendalltau(all_recs.old_score, all_recs.new_score)
        _log.info("correlation for user %s: %f", u, tau.correlation)
        correlations.loc[u] = tau.correlation

    _log.debug("correlations: %s", correlations)

    assert not (any(correlations.isnull()))
    assert all(correlations >= 0.5)


def test_als_predict_bad_item():
    algo = als.ImplicitMF(20, epochs=10)
    algo.fit(simple_ds)

    preds = algo.predict_for_user(10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_als_predict_bad_user():
    algo = als.ImplicitMF(20, epochs=10)
    algo.fit(simple_ds)

    preds = algo.predict_for_user(50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


def test_als_predict_no_user_features_basic(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    np.random.seed(45)
    u = np.random.choice(ml_ds.users, 1)[0]
    items = np.random.choice(ml_ds.items, 2)

    algo = als.ImplicitMF(5, epochs=10)
    algo.fit(ml_ds)
    preds = algo.predict_for_user(u, items)

    user_data = ml_ratings[ml_ratings.user == u]
    new_ratings = user_data.set_index("item")["rating"].copy()

    algo_no_user_features = als.ImplicitMF(5, epochs=10, save_user_features=False)
    algo_no_user_features.fit(ml_ds)
    preds_no_user_features = algo_no_user_features.predict_for_user(u, items, new_ratings)

    assert algo_no_user_features.user_features_ is None
    assert preds_no_user_features.values == approx(preds, abs=0.1)
    diffs = np.abs(preds - preds_no_user_features)
    assert all(diffs <= 0.1)


@lktu.wantjit
def test_als_train_large(ml_ds: Dataset):
    algo = als.ImplicitMF(20, epochs=20, use_ratings=False)
    algo.fit(ml_ds)

    assert algo.users_ is not None
    assert algo.user_features_ is not None
    assert len(algo.users_.index) == ml_ds.user_count
    assert len(algo.items_.index) == ml_ds.item_count
    assert algo.user_features_.shape == (ml_ds.user_count, 20)
    assert algo.item_features_.shape == (ml_ds.item_count, 20)


def test_als_save_load(tmp_path, ml_ds: Dataset):
    "Test saving and loading ALS models, and regularized training."
    algo = als.ImplicitMF(5, epochs=5, reg=(2, 1), use_ratings=False)
    algo.fit(ml_ds)
    assert algo.users_ is not None

    fn = tmp_path / "model.bpk"
    with fn.open("wb") as pf:
        pickle.dump(algo, pf, protocol=pickle.HIGHEST_PROTOCOL)

    with fn.open("rb") as pf:
        restored = pickle.load(pf)

    assert torch.all(restored.user_features_ == algo.user_features_)
    assert torch.all(restored.item_features_ == algo.item_features_)
    assert np.all(restored.items_.index == algo.items_.index)
    assert np.all(restored.users_.index == algo.users_.index)


@lktu.wantjit
def test_als_train_large_noratings(ml_ds: Dataset):
    algo = als.ImplicitMF(20, epochs=20)
    algo.fit(ml_ds)

    assert algo.users_ is not None
    assert algo.user_features_ is not None
    assert len(algo.users_.index) == ml_ds.user_count
    assert len(algo.items_.index) == ml_ds.item_count
    assert algo.user_features_.shape == (ml_ds.user_count, 20)
    assert algo.item_features_.shape == (ml_ds.item_count, 20)


@lktu.wantjit
def test_als_train_large_ratings(ml_ds):
    algo = als.ImplicitMF(20, epochs=20, use_ratings=True)
    algo.fit(ml_ds)

    assert algo.users_ is not None
    assert algo.user_features_ is not None
    assert len(algo.users_.index) == ml_ds.user_count
    assert len(algo.items_.index) == ml_ds.item_count
    assert algo.user_features_.shape == (ml_ds.user_count, 20)
    assert algo.item_features_.shape == (ml_ds.item_count, 20)


@mark.slow
@mark.eval
def test_als_implicit_batch_accuracy(ml_100k):
    import lenskit.crossfold as xf
    from lenskit import batch, topn

    ratings = load_movielens_df(lktu.ml_100k_zip)

    def eval(train, test):
        train = train.astype({"rating": np.float_})
        _log.info("training implicit MF")
        ials_algo = als.ImplicitMF(25, epochs=20)
        ials_algo = Recommender.adapt(ials_algo)
        ials_algo.fit(from_interactions_df(train))
        users = test.user.unique()
        _log.info("testing %d users", len(users))
        lu_recs = batch.recommend(ials_algo, users, 100, n_jobs=2)
        return lu_recs

    folds = list(xf.partition_users(ratings, 5, xf.SampleFrac(0.2)))
    test = pd.concat(te for (tr, te) in folds)
    recs = pd.concat((eval(train, test) for (train, test) in folds), ignore_index=True)

    _log.info("analyzing recommendations")
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, test)
    _log.info("nDCG for users is %.4f", results["ndcg"].mean())
    assert results["ndcg"].mean() > 0.28
