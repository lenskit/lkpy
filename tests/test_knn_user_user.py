from lenskit import matrix
import lenskit.algorithms.user_knn as knn

import logging
import os.path
import multiprocessing

import pandas as pd
import numpy as np

import pytest
from pytest import approx, mark

import lk_test_utils as lktu

_log = logging.getLogger(__name__)

ml_ratings = lktu.ml_pandas.renamed.ratings


@mark.slow
def test_uu_train():
    algo = knn.UserUser(30)
    model = algo.train(ml_ratings)

    # it should have computed correct means
    umeans = ml_ratings.groupby('user').rating.mean()
    mlmeans = model.user_stats['mean']
    umeans, mlmeans = umeans.align(mlmeans)
    assert mlmeans.values == approx(umeans.values)

    # we should be able to reconstruct rating values
    uir = ml_ratings.set_index(['user', 'item']).rating
    ui_rbdf = model.matrix.rename(columns={'rating': 'nrating'}).set_index(['user', 'item'])
    ui_rbdf = ui_rbdf.join(model.user_stats)
    ui_rbdf['rating'] = ui_rbdf['nrating'] * ui_rbdf['norm'] + ui_rbdf['mean']
    ui_rbdf['orig_rating'] = uir
    assert ui_rbdf.rating.values == approx(ui_rbdf.orig_rating.values)


def test_uu_predict_one():
    algo = knn.UserUser(30)
    model = algo.train(ml_ratings)

    preds = algo.predict(model, 4, [1016])
    assert len(preds) == 1
    assert preds.index == [1016]
    assert preds.values == approx([3.62221550680778])


def test_uu_predict_too_few():
    algo = knn.UserUser(30, min_nbrs=2)
    model = algo.train(ml_ratings)

    preds = algo.predict(model, 4, [2091])
    assert len(preds) == 1
    assert preds.index == [2091]
    assert all(preds.isna())


def test_uu_predict_too_few_blended():
    algo = knn.UserUser(30, min_nbrs=2)
    model = algo.train(ml_ratings)

    preds = algo.predict(model, 4, [1016, 2091])
    assert len(preds) == 2
    assert np.isnan(preds.loc[2091])
    assert preds.loc[1016] == approx(3.62221550680778)


def test_uu_predict_live_ratings():
    algo = knn.UserUser(30, min_nbrs=2)
    no4 = ml_ratings[ml_ratings.user != 4]
    model = algo.train(no4)

    ratings = ml_ratings[ml_ratings.user == 4].set_index('item').rating

    preds = algo.predict(model, 20381, [1016, 2091], ratings)
    assert len(preds) == 2
    assert np.isnan(preds.loc[2091])
    assert preds.loc[1016] == approx(3.62221550680778)


@mark.slow
def test_uu_save_load(tmp_path):
    tmp_path = lktu.norm_path(tmp_path)

    algo = knn.UserUser(30)
    _log.info('training model')
    original = algo.train(ml_ratings)

    fn = tmp_path / 'uu.model'
    _log.info('saving to %s', fn)
    algo.save_model(original, fn)

    _log.info('reloading model')
    model = algo.load_model(fn)
    _log.info('checking model')

    # it should have computed correct means
    umeans = ml_ratings.groupby('user').rating.mean()
    mlmeans = model.user_stats['mean']
    umeans, mlmeans = umeans.align(mlmeans)
    assert mlmeans.values == approx(umeans.values)

    # we should be able to reconstruct rating values
    uir = ml_ratings.set_index(['user', 'item']).rating
    ui_rbdf = model.matrix.rename(columns={'rating': 'nrating'}).set_index(['user', 'item'])
    ui_rbdf = ui_rbdf.join(model.user_stats)
    ui_rbdf['rating'] = ui_rbdf['nrating'] * ui_rbdf['norm'] + ui_rbdf['mean']
    ui_rbdf['orig_rating'] = uir
    assert ui_rbdf.rating.values == approx(ui_rbdf.orig_rating.values)


def test_uu_predict_unknown_empty():
    algo = knn.UserUser(30, min_nbrs=2)
    model = algo.train(ml_ratings)

    preds = algo.predict(model, -28018, [1016, 2091])
    assert len(preds) == 2
    assert all(preds.isna())


def __batch_eval(job):
    from lenskit import batch
    algo, train, test = job
    _log.info('running training')
    model = algo.train(train)
    _log.info('testing %d users', test.user.nunique())
    return batch.predict(lambda u, xs: algo.predict(model, u, xs), test)

@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_uu_batch_accuracy():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.load_ratings()

    uu_algo = knn.UserUser(30)
    algo = basic.Fallback(uu_algo, basic.Bias())

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    with multiprocessing.Pool() as pool:
        preds = pd.concat(pool.map(__batch_eval, ((algo, train, test) for (train, test) in folds)))
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.71, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.91, abs=0.05)
