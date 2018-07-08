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


def test_ii_train():
    algo = knn.ItemItem(30, save_nbrs=500)
    model = algo.train(simple_ratings)

    assert model is not None
    assert model.sim_matrix.index.name == 'item'
    assert list(model.sim_matrix.columns) == ['neighbor', 'similarity']
    assert all(model.sim_matrix.similarity.notna())
    assert all(model.sim_matrix.similarity > 0)

    assert 6 in model.sim_matrix.index
    nbr1 = model.sim_matrix.loc[[6], :]
    # 6 is a neighbor of 7
    assert (nbr1.neighbor == 7).sum() == 1

    svec = model.sim_matrix.set_index('neighbor', append=True).similarity
    assert all(svec.notna())
    assert all(svec > 0)
    # a little tolerance
    oob = svec[svec > 1 + 1.0e-6]
    assert len(oob) == 0

def test_ii_train_unbounded():
    algo = knn.ItemItem(30)
    model = algo.train(simple_ratings)

    assert model is not None
    assert model.sim_matrix.index.name == 'item'
    assert list(model.sim_matrix.columns) == ['neighbor', 'similarity']
    assert all(model.sim_matrix.similarity.notna())
    assert all(model.sim_matrix.similarity > 0)

    assert 6 in model.sim_matrix.index
    nbr1 = model.sim_matrix.loc[[6], :]
    # 6 is a neighbor of 7
    assert (nbr1.neighbor == 7).sum() == 1

    svec = model.sim_matrix.set_index('neighbor', append=True).similarity
    assert all(svec.notna())
    assert all(svec > 0)
    # a little tolerance
    oob = svec[svec > 1 + 1.0e-6]
    assert len(oob) == 0


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
    svec = model.sim_matrix.set_index('neighbor', append=True).similarity
    assert all(svec.notna())
    assert all(svec > 0)
    # a little tolerance
    oob = svec[svec > 1 + 1.0e-6]
    assert len(oob) == 0


@mark.slow
def test_ii_train_big_unbounded():
    algo = knn.ItemItem(30)
    model = algo.train(ml_ratings)

    assert model is not None
    assert model.sim_matrix.index.name == 'item'
    assert list(model.sim_matrix.columns) == ['neighbor', 'similarity']
    svec = model.sim_matrix.set_index('neighbor', append=True).similarity
    assert all(svec.notna())
    assert all(svec > 0)
    # a little tolerance
    oob = svec[svec > 1 + 1.0e-6]
    assert len(oob) == 0


@mark.slow
def test_ii_limited_model_is_subset():
    _log.info('training limited model')
    algo_lim = knn.ItemItem(30, save_nbrs=500)
    model_lim = algo_lim.train(ml_ratings)

    _log.info('training unbounded model')
    algo_ub = knn.ItemItem(30)
    model_ub = algo_ub.train(ml_ratings)

    _log.info('checking overall item set')
    items = model_ub.sim_matrix.index.unique()
    nitems = len(items)
    assert nitems == model_lim.sim_matrix.index.nunique()
    assert len(model_ub.sim_matrix.index.difference(model_lim.sim_matrix.index)) == 0

    _log.info('checking a sample of neighborhoods')
    for i in pd.Series(items).sample(20):
        ub_nbrs = model_ub.sim_matrix.loc[i]
        ub_nbrs = ub_nbrs.set_index('neighbor').similarity
        b_nbrs = model_lim.sim_matrix.loc[i]
        b_nbrs = b_nbrs.set_index('neighbor').similarity
        assert len(ub_nbrs) >= len(b_nbrs)
        assert len(b_nbrs) <= 500
        assert all(b_nbrs.index.isin(ub_nbrs.index))
        b_match, ub_match = b_nbrs.align(ub_nbrs, join='inner')
        assert all(b_match == b_nbrs)
        assert b_match.values == approx(ub_match.values)
        assert b_nbrs.max() == approx(ub_nbrs.max())
        if len(ub_nbrs) > 500:
            assert len(b_nbrs) == 500
            ub_shrink = ub_nbrs.nlargest(500)
            assert ub_shrink.min() == approx(b_nbrs.min())
            assert all(ub_shrink.index.isin(b_nbrs.index))


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

    with lktu.envvars(OMP_NUM_THREADS='1'):
        preds = batch.multi_predict(xf.partition_users(ratings, 5, xf.SampleFrac(0.2)),
                                    algo)
    # preds = pd.concat((eval(train, test)
    #                    for (train, test)
    #                    in xf.partition_users(ratings, 5, xf.SampleFrac(0.2))))
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.70, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.90, abs=0.05)
