import os.path
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
    model = algo.train(simple_df)

    assert model is not None
    assert model.global_bias == approx(simple_df.rating.mean())


def test_als_predict_basic():
    algo = als.BiasedMF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None
    assert model.global_bias == approx(simple_df.rating.mean())

    preds = algo.predict(model, 10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3] <= 5


def test_als_predict_bad_item():
    algo = als.BiasedMF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None
    assert model.global_bias == approx(simple_df.rating.mean())

    preds = algo.predict(model, 10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_als_predict_bad_user():
    algo = als.BiasedMF(20, iterations=10)
    model = algo.train(simple_df)

    assert model is not None
    assert model.global_bias == approx(simple_df.rating.mean())

    preds = algo.predict(model, 50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@mark.slow
def test_als_train_large():
    algo = als.BiasedMF(20, iterations=20)
    ratings = lktu.ml_pandas.renamed.ratings
    model = algo.train(ratings)

    assert model is not None
    assert model.global_bias == approx(ratings.rating.mean())

    icounts = ratings.groupby('item').rating.count()
    isums = ratings.groupby('item').rating.sum()
    is2 = isums - icounts * ratings.rating.mean()
    imeans = is2 / (icounts + 5)
    ibias = pd.Series(model.item_bias, index=model.item_index)
    imeans, ibias = imeans.align(ibias)
    assert ibias.values == approx(imeans.values)


@mark.slow
def test_als_save_load(tmpdir):
    mod_file = tmpdir / 'als.npz'
    algo = als.BiasedMF(20, iterations=20)
    ratings = lktu.ml_pandas.renamed.ratings
    model = algo.train(ratings)

    assert model is not None
    assert model.global_bias == approx(ratings.rating.mean())

    algo.save_model(model, mod_file)
    assert os.path.exists(mod_file)

    restored = algo.load_model(mod_file)
    assert restored.global_bias == model.global_bias
    assert np.all(restored.user_bias == model.user_bias)
    assert np.all(restored.item_bias == model.item_bias)
    assert np.all(restored.user_features == model.user_features)
    assert np.all(restored.item_features == model.item_features)
    assert np.all(restored.item_index == model.item_index)
    assert np.all(restored.user_index == model.user_index)


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_als_batch_accuracy():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.load_ratings()

    svd_algo = als.BiasedMF(25, iterations=20, damping=5)
    algo = basic.Fallback(svd_algo, basic.Bias(damping=5))

    def eval(train, test):
        _log.info('running training')
        model = algo.train(train)
        _log.info('testing %d users', test.user.nunique())
        return batch.predict(algo, test, model=model)

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    preds = pd.concat(eval(train, test) for (train, test) in folds)
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.73, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.91, abs=0.05)
