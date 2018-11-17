import os.path
import logging

from lenskit import topn, sharing
from lenskit.algorithms import als

import pandas as pd
import numpy as np

from pytest import approx, mark

import lk_test_utils as lktu

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


def test_als_basic_build():
    algo = als.ImplicitMF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None
    assert isinstance(model, als.MFModel)


def test_als_predict_basic():
    algo = als.ImplicitMF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None

    preds = algo.predict(model, 10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5


def test_als_predict_bad_item():
    algo = als.ImplicitMF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None

    preds = algo.predict(model, 10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_als_predict_bad_user():
    algo = als.ImplicitMF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None

    preds = algo.predict(model, 50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@mark.slow
def test_als_train_large():
    algo = als.ImplicitMF(20, iterations=20)
    ratings = lktu.ml_pandas.renamed.ratings
    model = algo.train(ratings)

    assert model is not None
    # FIXME Write more test assertions


def test_als_save_load(tmp_path):
    tmp_path = lktu.norm_path(tmp_path)
    mod_file = tmp_path / 'als.npz'
    algo = als.ImplicitMF(20, iterations=5)
    ratings = lktu.ml_pandas.renamed.ratings
    model = algo.train(ratings)

    assert model is not None

    algo.save_model(model, mod_file)
    assert mod_file.exists()

    restored = algo.load_model(mod_file)
    assert np.all(restored.user_features == model.user_features)
    assert np.all(restored.item_features == model.item_features)
    assert np.all(restored.item_index == model.item_index)
    assert np.all(restored.user_index == model.user_index)


def test_als_share():
    algo = als.ImplicitMF(20, iterations=5)
    ratings = lktu.ml_pandas.renamed.ratings
    model = algo.train(ratings)

    assert model is not None

    key = sharing.publish(model, algo)
    restored = sharing.resolve(key, algo)

    assert np.all(restored.user_features == model.user_features)
    assert np.all(restored.item_features == model.item_features)
    assert np.all(restored.item_index == model.item_index)
    assert np.all(restored.user_index == model.user_index)


@mark.slow
def test_als_train_large_noratings():
    algo = als.ImplicitMF(20, iterations=20)
    ratings = lktu.ml_pandas.renamed.ratings
    ratings = ratings.loc[:, ['user', 'item']]
    model = algo.train(ratings)

    assert model is not None
    # FIXME Write more test assertions


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_als_implicit_batch_accuracy():
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.topn as lm

    ratings = lktu.ml100k.load_ratings()

    algo = als.ImplicitMF(25, iterations=20)

    def eval(train, test):
        _log.info('running training')
        train['rating'] = train.rating.astype(np.float_)
        model = algo.train(train)
        users = test.user.unique()
        _log.info('testing %d users', len(users))
        candidates = topn.UnratedCandidates(train)
        recs = batch.recommend(algo, model, users, 100, candidates, test)
        return recs

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    recs = pd.concat(eval(train, test) for (train, test) in folds)

    _log.info('analyzing recommendations')
    ndcg = recs.groupby('user').rating.apply(lm.ndcg)
    _log.info('ndcg for users is %.4f', ndcg.mean())
    assert ndcg.mean() > 0
