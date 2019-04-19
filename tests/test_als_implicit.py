import logging
import pickle

from lenskit import topn, util
from lenskit.algorithms import als

import pandas as pd
import numpy as np

from pytest import mark

import lenskit.util.test as lktu

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


def test_als_basic_build():
    algo = als.ImplicitMF(20, iterations=10, progress=util.no_progress)
    algo.fit(simple_df)

    assert set(algo.user_index_) == set([10, 12, 13])
    assert set(algo.item_index_) == set([1, 2, 3])
    assert algo.user_features_.shape == (3, 20)
    assert algo.item_features_.shape == (3, 20)


def test_als_predict_basic():
    algo = als.ImplicitMF(20, iterations=10)
    algo.fit(simple_df)

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5


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
def test_als_train_large():
    algo = als.ImplicitMF(20, iterations=20)
    ratings = lktu.ml_test.ratings
    algo.fit(ratings)

    assert len(algo.user_index_) == ratings.user.nunique()
    assert len(algo.item_index_) == ratings.item.nunique()
    assert algo.user_features_.shape == (ratings.user.nunique(), 20)
    assert algo.item_features_.shape == (ratings.item.nunique(), 20)


def test_als_save_load():
    algo = als.ImplicitMF(20, iterations=5)
    ratings = lktu.ml_test.ratings
    algo.fit(ratings)

    mod = pickle.dumps(algo)
    _log.info('serialized to %d bytes', len(mod))

    restored = pickle.loads(mod)
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


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_als_implicit_batch_accuracy():
    import lenskit.crossfold as xf
    from lenskit import batch
    from lenskit import topn

    ratings = lktu.ml100k.ratings

    algo = als.ImplicitMF(25, iterations=20)

    def eval(train, test):
        _log.info('running training')
        train['rating'] = train.rating.astype(np.float_)
        algo.fit(train)
        users = test.user.unique()
        _log.info('testing %d users', len(users))
        candidates = topn.UnratedCandidates(train)
        recs = batch.recommend(algo, users, 100, candidates)
        return recs

    folds = list(xf.partition_users(ratings, 5, xf.SampleFrac(0.2)))
    test = pd.concat(te for (tr, te) in folds)
    recs = pd.concat(eval(train, test) for (train, test) in folds)

    _log.info('analyzing recommendations')
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, test)
    _log.info('nDCG for users is %.4f', results.ndcg.mean())
    assert results.ndcg.mean() > 0
