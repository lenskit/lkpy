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

from pytest import mark

import lenskit.util.test as lktu
from lenskit.algorithms import Recommender, als

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({"item": [1, 1, 2, 3], "user": [10, 12, 10, 13]})

simple_dfr = simple_df.assign(rating=[4.0, 3.0, 5.0, 2.0])


def test_als_basic_build():
    algo = als.ImplicitMF(20, epochs=10)
    algo.fit(simple_df)

    assert set(algo.user_index_) == set([10, 12, 13])
    assert set(algo.item_index_) == set([1, 2, 3])
    assert algo.user_features_.shape == (3, 20)
    assert algo.item_features_.shape == (3, 20)


def test_als_predict_basic():
    algo = als.ImplicitMF(20, epochs=10)
    algo.fit(simple_df)

    preds = algo.predict_for_user(10, [3])

    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5


def test_als_predict_basic_for_new_ratings():
    """Test ImplicitMF ability to support new ratings"""
    algo = als.ImplicitMF(20, epochs=10)
    algo.fit(simple_df)

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
    algo.fit(simple_dfr)

    preds = algo.predict_for_user(u, [i])

    new_u_id = 1
    new_ratings = pd.Series([4.0, 5.0], index=[1, 2])  # items as index and ratings as values

    new_preds = algo.predict_for_user(new_u_id, [i], new_ratings)
    assert abs(preds.loc[i] - new_preds.loc[i]) <= 0.1


def test_als_predict_for_new_users_with_new_ratings():
    """
    Test if ImplicitMF predictions using the same ratings for a new user
    is the same as a user in ml-latest-small dataset.
    The test is run for more than one user.
    """
    n_users = 3
    n_items = 2
    new_u_id = -1
    ratings = lktu.ml_test.ratings

    np.random.seed(45)
    users = np.random.choice(ratings.user.unique(), n_users)
    items = np.random.choice(ratings.item.unique(), n_items)

    algo = als.ImplicitMF(20, epochs=10, use_ratings=False)
    algo.fit(ratings)
    _log.debug("Items: " + str(items))

    for u in users:
        _log.debug(f"user: {u}")
        preds = algo.predict_for_user(u, items)
        upos = algo.user_index_.get_loc(u)

        # get the user's rating series
        user_data = ratings[ratings.user == u]
        new_ratings = user_data.set_index("item")["rating"].copy()

        nr_info = new_ratings.to_frame()
        ifs = algo.item_features_[algo.item_index_.get_indexer_for(nr_info.index), :]
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

        diffs = np.abs(preds.values - new_preds.values)
        assert all(diffs <= 0.1)


def test_als_recs_topn_for_new_users_with_new_ratings(rng):
    """
    Test if ImplicitMF topn recommendations using the same ratings for a new user
    is the same as a user in ml-latest-small dataset.
    The test is run for more than one user.
    """
    import scipy.stats as stats

    from lenskit.algorithms import basic

    n_users = 10
    new_u_id = -1
    ratings = lktu.ml_test.ratings

    users = rng.choice(np.unique(ratings.user), n_users)

    algo = als.ImplicitMF(20, epochs=10, use_ratings=True)
    rec_algo = basic.TopN(algo)
    rec_algo.fit(ratings)
    # _log.debug("Items: " + str(items))

    correlations = pd.Series(np.nan, index=users)
    for u in users:
        recs = rec_algo.recommend(u, 10)
        user_data = ratings[ratings.user == u]
        upos = algo.user_index_.get_loc(u)
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
    algo.fit(simple_df)

    preds = algo.predict_for_user(10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_als_predict_bad_user():
    algo = als.ImplicitMF(20, epochs=10)
    algo.fit(simple_df)

    preds = algo.predict_for_user(50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


def test_als_predict_no_user_features_basic():
    ratings = lktu.ml_test.ratings
    np.random.seed(45)
    u = np.random.choice(ratings.user.unique(), 1)[0]
    items = np.random.choice(ratings.item.unique(), 2)

    algo = als.ImplicitMF(5, epochs=10, use_ratings=True)
    algo.fit(ratings)
    preds = algo.predict_for_user(u, items)

    user_data = ratings[ratings.user == u]
    new_ratings = user_data.set_index("item")["rating"].copy()

    algo_no_user_features = als.ImplicitMF(5, epochs=10, save_user_features=False)
    algo_no_user_features.fit(ratings)
    preds_no_user_features = algo_no_user_features.predict_for_user(u, items, new_ratings)

    assert algo_no_user_features.user_features_ is None
    diffs = np.abs(preds.values - preds_no_user_features.values)
    assert all(diffs <= 0.1)


@lktu.wantjit
def test_als_train_large():
    algo = als.ImplicitMF(20, epochs=20, use_ratings=False)
    ratings = lktu.ml_test.ratings
    algo.fit(ratings)

    assert len(algo.user_index_) == ratings.user.nunique()
    assert len(algo.item_index_) == ratings.item.nunique()
    assert algo.user_features_.shape == (ratings.user.nunique(), 20)
    assert algo.item_features_.shape == (ratings.item.nunique(), 20)


def test_als_save_load(tmp_path):
    "Test saving and loading ALS models, and regularized training."
    algo = als.ImplicitMF(5, epochs=5, reg=(2, 1), use_ratings=False)
    ratings = lktu.ml_test.ratings
    algo.fit(ratings)

    fn = tmp_path / "model.bpk"
    with fn.open("wb") as pf:
        pickle.dump(algo, pf, protocol=pickle.HIGHEST_PROTOCOL)

    with fn.open("rb") as pf:
        restored = pickle.load(pf)

    assert torch.all(restored.user_features_ == algo.user_features_)
    assert torch.all(restored.item_features_ == algo.item_features_)
    assert np.all(restored.item_index_ == algo.item_index_)
    assert np.all(restored.user_index_ == algo.user_index_)


@lktu.wantjit
def test_als_train_large_noratings():
    algo = als.ImplicitMF(20, epochs=20)
    ratings = lktu.ml_test.ratings
    ratings = ratings.loc[:, ["user", "item"]]
    algo.fit(ratings)

    assert len(algo.user_index_) == ratings.user.nunique()
    assert len(algo.item_index_) == ratings.item.nunique()
    assert algo.user_features_.shape == (ratings.user.nunique(), 20)
    assert algo.item_features_.shape == (ratings.item.nunique(), 20)


@lktu.wantjit
def test_als_train_large_ratings():
    algo = als.ImplicitMF(20, epochs=20, use_ratings=True)
    ratings = lktu.ml_test.ratings
    algo.fit(ratings)

    assert len(algo.user_index_) == ratings.user.nunique()
    assert len(algo.item_index_) == ratings.item.nunique()
    assert algo.user_features_.shape == (ratings.user.nunique(), 20)
    assert algo.item_features_.shape == (ratings.item.nunique(), 20)


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason="ML100K data not present")
def test_als_implicit_batch_accuracy():
    import lenskit.crossfold as xf
    from lenskit import batch, topn

    ratings = lktu.ml100k.ratings

    def eval(train, test):
        train = train.astype({"rating": np.float_})
        _log.info("training implicit MF")
        ials_algo = als.ImplicitMF(25, epochs=20)
        ials_algo = Recommender.adapt(ials_algo)
        ials_algo.fit(train)
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
