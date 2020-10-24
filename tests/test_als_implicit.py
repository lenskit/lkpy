import logging
import pickle

from lenskit import util
from lenskit.algorithms import als

import pandas as pd
import numpy as np
from scipy import stats
import binpickle

from pytest import mark, approx

import lenskit.util.test as lktu
from lenskit.algorithms import Recommender
from lenskit.util import Stopwatch

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


methods = mark.parametrize('m', ['lu', 'cg'])


@methods
def test_als_basic_build(m):
    algo = als.ImplicitMF(20, iterations=10, progress=util.no_progress, method=m)
    algo.fit(simple_df)

    assert set(algo.user_index_) == set([10, 12, 13])
    assert set(algo.item_index_) == set([1, 2, 3])
    assert algo.user_features_.shape == (3, 20)
    assert algo.item_features_.shape == (3, 20)


def test_als_predict_basic():
    algo = als.ImplicitMF(20, iterations=10)
    algo.fit(simple_df)

    preds = algo.predict_for_user(10, [3])
    print(preds)
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5

def test_als_predict_basic_for_new_ratings():
    algo = als.ImplicitMF(20, iterations=10)
    algo.fit(simple_df)

    new_ratings = pd.Series([4.0, 5.0], index=[1, 2]) # items as index and ratings as values

    preds = algo.predict_for_user(15, [3], new_ratings)

    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5

def test_als_predict_basic_for_new_user_with_new_ratings():
    u = 10
    i = 3

    algo = als.ImplicitMF(20, iterations=10)
    algo.fit(simple_df)

    preds = algo.predict_for_user(u, [i])

    new_u_id = 1
    new_ratings = pd.Series([4.0, 5.0], index=[1, 2]) # items as index and ratings as values

    new_preds = algo.predict_for_user(new_u_id, [i], new_ratings)
    assert abs(preds.loc[i] - new_preds.loc[i]) <= 0.1

def test_als_predict_for_new_users_with_new_ratings():
    n_users = 3
    n_items = 2
    new_u_id = -1
    ratings = lktu.ml_test.ratings

    np.random.seed(45)
    users = np.random.choice(ratings.user.unique(), n_users)
    items = np.random.choice(ratings.item.unique(), n_items)

    algo = als.ImplicitMF(20, iterations=10, method="lu")
    algo.fit(ratings)
    _log.debug("Items: " + str(items))

    for u in users:
        _log.debug(f"user: {u}")
        preds = algo.predict_for_user(u, items)

        user_data = ratings[ratings.user == u]

        _log.debug("user_features from fit: " + str(algo.user_features_[algo.user_index_.get_loc(u), :]))

        new_ratings = pd.Series(user_data.rating.to_numpy(), index=user_data.item) # items as index and ratings as values
        new_preds = algo.predict_for_user(new_u_id, items, new_ratings)

        _log.debug("preds: " + str(preds.values))
        _log.debug("new_preds: " + str(new_preds.values))
        _log.debug("------------")

        diffs = abs(preds.values - new_preds.values)
        assert all(i <= 0.1 for i in diffs) == True

# TODO: Add a test to check if ImplicitMF with new ratings gives the same top 10 recommendations as the current one.

def test_als_predict_bad_item():
    algo = als.ImplicitMF(20, iterations=10)
    algo.fit(simple_df)

    preds = algo.predict_for_user(10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_als_predict_bad_user():
    algo = als.ImplicitMF(20, iterations=10)
    algo.fit(simple_df)

    preds = algo.predict_for_user(50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])

@lktu.wantjit
@methods
def test_als_train_large(m):
    algo = als.ImplicitMF(20, iterations=20, method=m)
    ratings = lktu.ml_test.ratings
    algo.fit(ratings)

    assert len(algo.user_index_) == ratings.user.nunique()
    assert len(algo.item_index_) == ratings.item.nunique()
    assert algo.user_features_.shape == (ratings.user.nunique(), 20)
    assert algo.item_features_.shape == (ratings.item.nunique(), 20)


def test_als_save_load(tmp_path):
    "Test saving and loading ALS models, and regularized training."
    algo = als.ImplicitMF(5, iterations=5, reg=(2, 1))
    ratings = lktu.ml_test.ratings
    algo.fit(ratings)

    fn = tmp_path / 'model.bpk'
    binpickle.dump(algo, fn, codec=None)

    restored = binpickle.load(fn)
    assert np.all(restored.user_features_ == algo.user_features_)
    assert np.all(restored.item_features_ == algo.item_features_)
    assert np.all(restored.item_index_ == algo.item_index_)
    assert np.all(restored.user_index_ == algo.user_index_)


@lktu.wantjit
def test_als_train_large_noratings():
    algo = als.ImplicitMF(20, iterations=20)
    ratings = lktu.ml_test.ratings
    ratings = ratings.loc[:, ['user', 'item']]
    algo.fit(ratings)

    assert len(algo.user_index_) == ratings.user.nunique()
    assert len(algo.item_index_) == ratings.item.nunique()
    assert algo.user_features_.shape == (ratings.user.nunique(), 20)
    assert algo.item_features_.shape == (ratings.item.nunique(), 20)


@lktu.wantjit
@mark.slow
def test_als_method_match():
    lu = als.ImplicitMF(20, iterations=15, method='lu', rng_spec=42)
    cg = als.ImplicitMF(20, iterations=15, method='cg', rng_spec=42)

    ratings = lktu.ml_test.ratings

    timer = Stopwatch()
    lu.fit(ratings)
    timer.stop()
    _log.info('fit with LU solver in %s', timer)

    timer = Stopwatch()
    cg.fit(ratings)
    timer.stop()
    _log.info('fit with CG solver in %s', timer)

    preds = []

    rng = util.rng(42, legacy=True)
    for u in rng.choice(ratings.user.unique(), 10, replace=False):
        items = rng.choice(ratings.item.unique(), 15, replace=False)
        lu_preds = lu.predict_for_user(u, items)
        cd_preds = cg.predict_for_user(u, items)
        diff = lu_preds - cd_preds
        adiff = np.abs(diff)
        _log.info('user %s diffs: L2 = %f, min = %f, med = %f, max = %f, 90%% = %f', u,
                  np.linalg.norm(diff, 2),
                  np.min(adiff), np.median(adiff), np.max(adiff), np.quantile(adiff, 0.9))

        preds.append(pd.DataFrame({
            'user': u,
            'item': items,
            'lu': lu_preds,
            'cg': cd_preds,
            'adiff': adiff
        }))
        _log.info('user %s tau: %s', u, stats.kendalltau(lu_preds, cd_preds))

    preds = pd.concat(preds, ignore_index=True)
    _log.info('LU preds:\n%s', preds.lu.describe())
    _log.info('CD preds:\n%s', preds.cg.describe())
    _log.info('overall differences:\n%s', preds.adiff.describe())
    # there are differences. our check: the 90% are reasonable
    assert np.quantile(adiff, 0.9) < 0.5


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_als_implicit_batch_accuracy():
    import lenskit.crossfold as xf
    from lenskit import batch
    from lenskit import topn

    ratings = lktu.ml100k.ratings

    def eval(train, test):
        train['rating'] = train.rating.astype(np.float_)
        _log.info('training CG')
        cg_algo = als.ImplicitMF(25, iterations=20, method='cg')
        cg_algo = Recommender.adapt(cg_algo)
        cg_algo.fit(train)
        _log.info('training LU')
        lu_algo = als.ImplicitMF(25, iterations=20, method='lu')
        lu_algo = Recommender.adapt(lu_algo)
        lu_algo.fit(train)
        users = test.user.unique()
        _log.info('testing %d users', len(users))
        cg_recs = batch.recommend(cg_algo, users, 100, n_jobs=2)
        lu_recs = batch.recommend(lu_algo, users, 100, n_jobs=2)
        return pd.concat({'CG': cg_recs, 'LU': lu_recs}, names=['Method']).reset_index('Method')

    folds = list(xf.partition_users(ratings, 5, xf.SampleFrac(0.2)))
    test = pd.concat(te for (tr, te) in folds)
    recs = pd.concat((eval(train, test) for (train, test) in folds), ignore_index=True)

    _log.info('analyzing recommendations')
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, test)
    results = results.groupby('Method')['ndcg'].mean()
    _log.info('LU nDCG for users is %.4f', results.loc['LU'].mean())
    _log.info('CG nDCG for users is %.4f', results.loc['CG'].mean())
    assert all(results > 0.28)
    assert results.loc['LU'] == approx(results.loc['CG'], rel=0.05)
