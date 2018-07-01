import lenskit.algorithms.item_knn as knn
import lenskit.algorithms._item_knn as _knn

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


def test_sparse_dot_empty():
    keys = np.arange(0, 0, dtype=np.int_)
    values = np.array([], dtype=np.float_)
    assert _knn.sparse_dot(keys, values, keys, values) == 0.0


def test_sparse_dot_disjoint():
    ks1 = np.arange(0, 2, dtype=np.int_)
    assert len(ks1) == 2
    ks2 = np.arange(2, 4, dtype=np.int_)
    assert len(ks2) == 2
    values = np.array([1.0, 2.0], dtype=np.float_)
    assert _knn.sparse_dot(ks1, values, ks2, values) == 0.0


def test_sparse_dot_two():
    ks1 = np.arange(0, 2, dtype=np.int_)
    assert len(ks1) == 2
    ks2 = np.arange(0, 2, dtype=np.int_)
    assert len(ks2) == 2
    vs1 = np.array([1.0, 2.0], dtype=np.float_)
    vs2 = np.array([0.5, -0.5], dtype=np.float_)
    assert _knn.sparse_dot(ks1, vs1, ks2, vs2) == approx(-0.5)


def test_sparse_dot_subset():
    ks1 = np.arange(0, 4, dtype=np.int_)
    ks2 = np.arange(1, 3, dtype=np.int_)
    vs1 = np.array([100, 1.0, 2.0, 50], dtype=np.float_)
    vs2 = np.array([0.5, -0.5], dtype=np.float_)
    assert _knn.sparse_dot(ks1, vs1, ks2, vs2) == approx(-0.5)


def test_sparse_dot_skip():
    ks1 = np.arange(0, 4, dtype=np.int_)
    ks2 = np.array([0, 2], dtype=np.int_)
    vs1 = np.array([1.0, 100, 2.0, 50], dtype=np.float_)
    vs2 = np.array([0.3, -0.5], dtype=np.float_)
    assert _knn.sparse_dot(ks1, vs1, ks2, vs2) == approx(-0.7)
    assert _knn.sparse_dot(ks2, vs2, ks1, vs1) == approx(-0.7)


def test_ii_train():
    algo = knn.ItemItem(30, save_nbrs=500)
    model = algo.train(simple_ratings)

    assert model is not None
    assert model.sim_matrix.index.name == 'item'
    assert list(model.sim_matrix.columns) == ['neighbor', 'similarity']
    assert all(model.sim_matrix.similarity.notna())
    assert all(model.sim_matrix.similarity > 0)

    nbr1 = model.sim_matrix.loc[[6], :]
    # 6 is a neighbor of 7
    assert (nbr1.neighbor == 7).sum() == 1


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
    from lenskit.algorithms import baselines, basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    if not os.path.exists('ml-100k/u.data'):
        raise pytest.skip()

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    uu_algo = knn.ItemItem(30)
    algo = basic.Fallback(uu_algo, baselines.Bias())

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
