import logging

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
    algo = als.BiasedMF(20, iterations=10)
    algo.fit(simple_df)

    assert algo.global_bias_ == approx(simple_df.rating.mean())
    assert set(algo.user_index_) == set([10, 12, 13])
    assert set(algo.item_index_) == set([1, 2, 3])
    assert algo.user_features_.shape == (3, 20)
    assert algo.item_features_.shape == (3, 20)

    assert algo.n_features == 20
    assert algo.n_users == 3
    assert algo.n_items == 3


def test_als_no_bias():
    algo = als.BiasedMF(20, iterations=10, bias=None)
    algo.fit(simple_df)
    assert algo.bias is None

    assert algo.global_bias_ == 0
    assert algo.item_bias_ is None
    assert algo.user_bias_ is None
    assert set(algo.user_index_) == set([10, 12, 13])
    assert set(algo.item_index_) == set([1, 2, 3])
    assert algo.user_features_.shape == (3, 20)
    assert algo.item_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1


def test_als_predict_basic():
    algo = als.BiasedMF(20, iterations=10)
    algo.fit(simple_df)

    assert algo.global_bias_ == approx(simple_df.rating.mean())

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= -0.1
    assert preds.loc[3] <= 5.1


def test_als_predict_bad_item():
    algo = als.BiasedMF(20, iterations=10)
    algo.fit(simple_df)

    assert algo.global_bias_ == approx(simple_df.rating.mean())

    preds = algo.predict_for_user(10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_als_predict_bad_user():
    algo = als.BiasedMF(20, iterations=10)
    algo.fit(simple_df)

    assert algo.global_bias_ == approx(simple_df.rating.mean())

    preds = algo.predict_for_user(50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@lktu.wantjit
def test_als_train_large():
    algo = als.BiasedMF(20, iterations=10)
    ratings = lktu.ml_pandas.renamed.ratings
    algo.fit(ratings)

    assert algo.global_bias_ == approx(ratings.rating.mean())
    assert algo.n_features == 20
    assert algo.n_items == ratings.item.nunique()
    assert algo.n_users == ratings.user.nunique()

    icounts = ratings.groupby('item').rating.count()
    isums = ratings.groupby('item').rating.sum()
    is2 = isums - icounts * ratings.rating.mean()
    imeans = is2 / (icounts + 5)
    ibias = pd.Series(algo.item_bias_, index=algo.item_index_)
    imeans, ibias = imeans.align(ibias)
    assert ibias.values == approx(imeans.values)


def test_als_save_load(tmp_path):
    tmp_path = lktu.norm_path(tmp_path)
    mod_file = tmp_path / 'als.npz'
    original = als.BiasedMF(20, iterations=5)
    ratings = lktu.ml_pandas.renamed.ratings
    original.fit(ratings)

    assert original.global_bias_ == approx(ratings.rating.mean())

    original.save(mod_file)
    assert mod_file.exists()

    algo = als.BiasedMF(20)
    algo.load(mod_file)
    assert algo.global_bias_ == original.global_bias_
    assert np.all(algo.user_bias_ == original.user_bias_)
    assert np.all(algo.item_bias_ == original.item_bias_)
    assert np.all(algo.user_features_ == original.user_features_)
    assert np.all(algo.item_features_ == original.item_features_)
    assert np.all(algo.item_index_ == original.item_index_)
    assert np.all(algo.user_index_ == original.user_index_)


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_als_batch_accuracy():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.load_ratings()

    svd_algo = als.BiasedMF(25, iterations=20, damping=5)
    algo = basic.Fallback(svd_algo, basic.Bias(damping=5))

    def eval(train, test):
        _log.info('running training')
        algo.fit(train)
        _log.info('testing %d users', test.user.nunique())
        return test.assign(prediction=algo.predict(test))

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    preds = pd.concat(eval(train, test) for (train, test) in folds)
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.73, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.91, abs=0.05)
