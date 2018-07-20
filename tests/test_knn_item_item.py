import lenskit.algorithms.item_knn as knn

import logging
import os.path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

import pytest
from pytest import approx, mark

import lk_test_utils as lktu
from lk_test_utils import tmpdir

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

    # 6 is a neighbor of 7
    six, seven = model.items.get_indexer([6, 7])
    assert model.sim_matrix[six, seven] > 0

    assert all(np.logical_not(np.isnan(model.sim_matrix.data)))
    assert all(model.sim_matrix.data > 0)
    # a little tolerance
    assert all(model.sim_matrix.data < 1 + 1.0e-6)


def test_ii_train_unbounded():
    algo = knn.ItemItem(30)
    model = algo.train(simple_ratings)

    assert model is not None

    assert all(np.logical_not(np.isnan(model.sim_matrix.data)))
    assert all(model.sim_matrix.data > 0)
    # a little tolerance
    assert all(model.sim_matrix.data < 1 + 1.0e-6)

    # 6 is a neighbor of 7
    six, seven = model.items.get_indexer([6, 7])
    assert model.sim_matrix[six, seven] > 0


def test_ii_simple_predict():
    algo = knn.ItemItem(30, save_nbrs=500)
    model = algo.train(simple_ratings)

    res = algo.predict(model, 3, [6])
    assert res is not None
    assert len(res) == 1
    assert 6 in res.index
    assert not np.isnan(res.loc[6])


@mark.slow
@mark.skip("redundant with large_models")
def test_ii_train_big():
    "Simple tests for bounded models"
    algo = knn.ItemItem(30, save_nbrs=500)
    model = algo.train(ml_ratings)

    assert model is not None

    assert all(np.logical_not(np.isnan(model.sim_matrix.data)))
    assert all(model.sim_matrix.data > 0)
    # a little tolerance
    assert all(model.sim_matrix.data < 1 + 1.0e-6)

    assert model.counts.sum() == model.sim_matrix.nnz

    means = ml_ratings.groupby('item').rating.mean()
    assert means[model.items].values == approx(model.means)


@mark.slow
@mark.skip("redundant with large_models")
def test_ii_train_big_unbounded():
    "Simple tests for unbounded models"
    algo = knn.ItemItem(30)
    model = algo.train(ml_ratings)

    assert model is not None

    assert all(np.logical_not(np.isnan(model.sim_matrix.data)))
    assert all(model.sim_matrix.data > 0)
    # a little tolerance
    assert all(model.sim_matrix.data < 1 + 1.0e-6)

    assert model.counts.sum() == model.sim_matrix.nnz

    means = ml_ratings.groupby('item').rating.mean()
    assert means[model.items].values == approx(model.means)


@mark.slow
def test_ii_large_models():
    "Several tests of large trained I-I models"
    exec = ThreadPoolExecutor()
    _log.info('kicking off limited model train')
    algo_lim = knn.ItemItem(30, save_nbrs=500)
    model_lim = exec.submit(algo_lim.train, ml_ratings)

    _log.info('kicking off unbounded model')
    algo_ub = knn.ItemItem(30)
    model_ub = exec.submit(algo_ub.train, ml_ratings)

    model_lim = model_lim.result()
    _log.info('completed limited train')

    assert all(np.logical_not(np.isnan(model_lim.sim_matrix.data)))
    assert all(model_lim.sim_matrix.data > 0)
    # a little tolerance
    assert all(model_lim.sim_matrix.data < 1 + 1.0e-6)

    means = ml_ratings.groupby('item').rating.mean()
    assert means[model_lim.items].values == approx(model_lim.means)

    model_ub = model_ub.result()
    _log.info('completed unbounded train')

    assert all(np.logical_not(np.isnan(model_ub.sim_matrix.data)))
    assert all(model_ub.sim_matrix.data > 0)
    # a little tolerance
    assert all(model_ub.sim_matrix.data < 1 + 1.0e-6)

    means = ml_ratings.groupby('item').rating.mean()
    assert means[model_ub.items].values == approx(model_ub.means)

    mc_rates = ml_ratings.set_index('item')\
                         .join(pd.DataFrame({'item_mean': means}))\
                         .assign(rating=lambda df: df.rating - df.item_mean)

    _log.info('checking a sample of neighborhoods')
    items = pd.Series(model_ub.items)
    items = items[model_ub.counts > 0]
    for i in items.sample(50):
        ipos = model_ub.items.get_loc(i)
        _log.debug('checking item %d at position %d', i, ipos)
        assert ipos == model_lim.items.get_loc(i)
        irates = mc_rates.loc[[i], :].set_index('user').rating

        ub_row = model_ub.sim_matrix.getrow(ipos)
        b_row = model_lim.sim_matrix.getrow(ipos)
        assert b_row.nnz <= 500
        assert all(pd.Series(b_row.indices).isin(ub_row.indices))

        # it should be sorted !
        # check this by diffing the row values, and make sure they're negative
        assert all(np.diff(b_row.data) < 1.0e-6)
        assert all(np.diff(ub_row.data) < 1.0e-6)

        # spot-check some similarities
        for n in pd.Series(ub_row.indices).sample(min(10, len(ub_row.indices))):
            n_id = model_ub.items[n]
            n_rates = mc_rates.loc[n_id, :].set_index('user').rating
            ir, nr = irates.align(n_rates, fill_value=0)
            cor = ir.corr(nr)
            assert model_ub.sim_matrix[ipos, n] == approx(cor)

        # short rows are equal
        if b_row.nnz < 500:
            _log.debug('short row of length %d', b_row.nnz)
            assert b_row.nnz == ub_row.nnz
            ub_row.sort_indices()
            b_row.sort_indices()
            assert b_row.data == approx(ub_row.data)
            continue

        # row is truncated - check that truncation is correct
        ub_nbrs = pd.Series(ub_row.data, model_ub.items[ub_row.indices])
        b_nbrs = pd.Series(b_row.data, model_lim.items[b_row.indices])

        assert len(ub_nbrs) >= len(b_nbrs)
        assert len(b_nbrs) <= 500
        assert all(b_nbrs.index.isin(ub_nbrs.index))
        # the similarities should be equal!
        b_match, ub_match = b_nbrs.align(ub_nbrs, join='inner')
        assert all(b_match == b_nbrs)
        assert b_match.values == approx(ub_match.values)
        assert b_nbrs.max() == approx(ub_nbrs.max())
        if len(ub_nbrs) > 500:
            assert len(b_nbrs) == 500
            ub_shrink = ub_nbrs.nlargest(500)
            # the minimums should be equal
            assert ub_shrink.min() == approx(b_nbrs.min())
            # everything above minimum value should be the same set of items
            ubs_except_min = ub_shrink[ub_shrink > b_nbrs.min()]
            assert all(ubs_except_min.index.isin(b_nbrs.index))


@mark.slow
def test_ii_save_load(tmpdir):
    "Save and load a model"
    algo = knn.ItemItem(30, save_nbrs=500)
    _log.info('building model')
    original = algo.train(ml_ratings)

    fn = os.path.join(tmpdir, 'ii.mod')
    _log.info('saving model to %s', fn)
    algo.save_model(original, fn)
    _log.info('reloading model')
    model = algo.load_model(fn)
    _log.info('checking model')

    assert model is not None
    assert model is not original

    assert all(np.logical_not(np.isnan(model.sim_matrix.data)))
    assert all(model.sim_matrix.data > 0)
    # a little tolerance
    assert all(model.sim_matrix.data < 1 + 1.0e-6)

    assert all(model.counts == original.counts)
    assert model.counts.sum() == model.sim_matrix.nnz
    assert model.sim_matrix.nnz == original.sim_matrix.nnz
    assert all(model.sim_matrix.indptr == original.sim_matrix.indptr)
    assert all(model.sim_matrix.indices == original.sim_matrix.indices)
    assert model.sim_matrix.data == approx(original.sim_matrix.data)

    means = ml_ratings.groupby('item').rating.mean()
    assert means[model.items].values == approx(original.means)

     items = pd.Series(model.items)
     items = items[model.counts > 0]
     for i in items.sample(50):
         ipos = model.items.get_loc(i)
         _log.debug('checking item %d at position %d', i, ipos)

         row = model.sim_matrix.getrow(ipos)

         # it should be sorted !
         # check this by diffing the row values, and make sure they're negative
         assert all(np.diff(row.data) < 1.0e-6)


@mark.slow
@mark.eval
def test_ii_batch_accuracy():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    if not os.path.exists('ml-100k/u.data'):
        raise pytest.skip()

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    uu_algo = knn.ItemItem(30)
    algo = basic.Fallback(uu_algo, basic.Bias())

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
