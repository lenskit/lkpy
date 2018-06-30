import lenskit.algorithms.item_knn as knn

import logging
import os.path

import pandas as pd
import numpy as np

import pytest
from pytest import approx, mark

import lk_test_utils as lktu

_log = logging.getLogger(__name__)

ml_ratings = lktu.ml_pandas.renamed.ratings
simple_ratings = pd.DataFrame.from_records([
    (1, 6, 4.0),
    (2, 6, 2.0),
    (1, 7, 3.0),
    (2, 7, 2.0),
    (3, 7, 5.0),
    (4, 7, 2.0),
    (1, 8, 3.0),
    (2, 8, 4.0),
    (3, 8, 3.0),
    (4, 8, 2.0),
    (5, 8, 3.0),
    (6, 8, 2.0),
    (1, 9, 3.0),
    (3, 9, 4.0)
], columns=['user', 'item', 'rating'])


def test_ii_train():
    algo = knn.ItemItem(30, save_nbrs=500)
    model = algo.train(simple_ratings)

    assert model is not None
    assert model.sim_matrix.index.name == 'item'
    assert list(model.sim_matrix.columns) == ['neighbor', 'similarity']


def test_ii_simple_predict():
    algo = knn.ItemItem(30, save_nbrs=500)
    model = algo.train(simple_ratings)

    res = algo.predict(model, 3, [6])
    assert res is not None
    assert len(res) == 1
    assert 6 in res.index
    assert not np.isnan(res.loc[6])


@mark.slow
def test_ii_train_big():
    algo = knn.ItemItem(30, save_nbrs=500)
    model = algo.train(ml_ratings)

    assert model is not None
    assert model.sim_matrix.index.name == 'item'
    assert list(model.sim_matrix.columns) == ['neighbor', 'similarity']


@mark.slow
def test_ii_batch_accuracy():
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    if not os.path.exists('ml-100k/u.data'):
        raise pytest.skip()

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    algo = knn.ItemItem(30)

    def eval(train, test):
        _log.info('running training')
        model = algo.train(train)
        _log.info('testing %d users', test.user.nunique())
        return batch.predict(lambda u, xs: algo.predict(model, u, xs), test)

    preds = batch.multi_predict(xf.partition_users(ratings, 5, xf.SampleFrac(0.2)),
                                algo)
    # preds = pd.concat((eval(train, test)
    #                    for (train, test)
    #                    in xf.partition_users(ratings, 5, xf.SampleFrac(0.2))))
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.70, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.90, abs=0.05)
