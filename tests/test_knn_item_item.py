import lenskit.algorithms.item_knn as knn
from lenskit import matrix as lm

from pathlib import Path
import logging
import os.path

import pandas as pd
import numpy as np
from scipy import linalg as la
from scipy import sparse as sps

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
    assert isinstance(model.items, pd.Index)
    assert isinstance(model.means, np.ndarray)
    assert isinstance(model.counts, np.ndarray)
    matrix = lm.csr_to_scipy(model.sim_matrix)

    # 6 is a neighbor of 7
    six, seven = model.items.get_indexer([6, 7])
    _log.info('six: %d', six)
    _log.info('seven: %d', seven)
    _log.info('matrix: %s', model.sim_matrix)
    assert matrix[six, seven] > 0
    # and has the correct score
    six_v = simple_ratings[simple_ratings.item == 6].set_index('user').rating
    six_v = six_v - six_v.mean()
    seven_v = simple_ratings[simple_ratings.item == 7].set_index('user').rating
    seven_v = seven_v - seven_v.mean()
    denom = la.norm(six_v.values) * la.norm(seven_v.values)
    six_v, seven_v = six_v.align(seven_v, join='inner')
    num = six_v.dot(seven_v)
    assert matrix[six, seven] == approx(num / denom, 0.01)

    assert all(np.logical_not(np.isnan(model.sim_matrix.values)))
    assert all(model.sim_matrix.values > 0)
    # a little tolerance
    assert all(model.sim_matrix.values < 1 + 1.0e-6)


def test_ii_train_unbounded():
    algo = knn.ItemItem(30)
    model = algo.train(simple_ratings)

    assert model is not None

    assert all(np.logical_not(np.isnan(model.sim_matrix.values)))
    assert all(model.sim_matrix.values > 0)
    # a little tolerance
    assert all(model.sim_matrix.values < 1 + 1.0e-6)

    # 6 is a neighbor of 7
    matrix = lm.csr_to_scipy(model.sim_matrix)
    six, seven = model.items.get_indexer([6, 7])
    assert matrix[six, seven] > 0

    # and has the correct score
    six_v = simple_ratings[simple_ratings.item == 6].set_index('user').rating
    six_v = six_v - six_v.mean()
    seven_v = simple_ratings[simple_ratings.item == 7].set_index('user').rating
    seven_v = seven_v - seven_v.mean()
    denom = la.norm(six_v.values) * la.norm(seven_v.values)
    six_v, seven_v = six_v.align(seven_v, join='inner')
    num = six_v.dot(seven_v)
    assert matrix[six, seven] == approx(num / denom, 0.01)


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

    assert all(np.logical_not(np.isnan(model.sim_matrix.values)))
    assert all(model.sim_matrix.values > 0)
    # a little tolerance
    assert all(model.sim_matrix.values < 1 + 1.0e-6)

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

    assert all(np.logical_not(np.isnan(model.sim_matrix.values)))
    assert all(model.sim_matrix.values > 0)
    # a little tolerance
    assert all(model.sim_matrix.values < 1 + 1.0e-6)

    assert model.counts.sum() == model.sim_matrix.nnz

    means = ml_ratings.groupby('item').rating.mean()
    assert means[model.items].values == approx(model.means)


@mark.slow
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_ii_train_ml100k(tmp_path):
    "Test an unbounded model on ML-100K"
    tmp_path = lktu.norm_path(tmp_path)

    ratings = lktu.ml100k.load_ratings()
    algo = knn.ItemItem(30)
    _log.info('training model')
    model = algo.train(ratings)

    _log.info('testing model')
    assert model is not None

    assert all(np.logical_not(np.isnan(model.sim_matrix.values)))
    assert all(model.sim_matrix.values > 0)

    # a little tolerance
    assert all(model.sim_matrix.values < 1 + 1.0e-6)

    assert model.counts.sum() == model.sim_matrix.nnz

    means = ratings.groupby('item').rating.mean()
    assert means[model.items].values == approx(model.means)

    # save
    fn = tmp_path / 'ii.mod'
    _log.info('saving model to %s', fn)
    algo.save_model(model, fn)
    _log.info('reloading model')
    restored = algo.load_model(fn)
    assert restored is not None and restored is not model
    assert all(restored.sim_matrix.values > 0)

    r_mat = restored.sim_matrix
    o_mat = model.sim_matrix

    assert all(r_mat.rowptrs == o_mat.rowptrs)

    for i in range(len(restored.items)):
        sp = r_mat.rowptrs[i]
        ep = r_mat.rowptrs[i + 1]

        # everything is in decreasing order
        assert all(np.diff(r_mat.values[sp:ep]) <= 0)
        assert all(r_mat.values[sp:ep] == o_mat.values[sp:ep])


@mark.slow
def test_ii_large_models():
    "Several tests of large trained I-I models"
    _log.info('training limited model')
    MODEL_SIZE = 100
    algo_lim = knn.ItemItem(30, save_nbrs=MODEL_SIZE)
    model_lim = algo_lim.train(ml_ratings)

    _log.info('training unbounded model')
    algo_ub = knn.ItemItem(30)
    model_ub = algo_ub.train(ml_ratings)

    _log.info('testing models')
    assert all(np.logical_not(np.isnan(model_lim.sim_matrix.values)))
    assert all(model_lim.sim_matrix.values > 0)
    # a little tolerance
    assert all(model_lim.sim_matrix.values < 1 + 1.0e-6)

    means = ml_ratings.groupby('item').rating.mean()
    assert means[model_lim.items].values == approx(model_lim.means)

    assert all(np.logical_not(np.isnan(model_ub.sim_matrix.values)))
    assert all(model_ub.sim_matrix.values > 0)
    # a little tolerance
    assert all(model_ub.sim_matrix.values < 1 + 1.0e-6)

    means = ml_ratings.groupby('item').rating.mean()
    assert means[model_ub.items].values == approx(model_ub.means)

    mc_rates = ml_ratings.set_index('item')\
                         .join(pd.DataFrame({'item_mean': means}))\
                         .assign(rating=lambda df: df.rating - df.item_mean)

    mat_lim = lm.csr_to_scipy(model_lim.sim_matrix)
    mat_ub = lm.csr_to_scipy(model_ub.sim_matrix)

    _log.info('checking a sample of neighborhoods')
    items = pd.Series(model_ub.items)
    items = items[model_ub.counts > 0]
    for i in items.sample(50):
        ipos = model_ub.items.get_loc(i)
        _log.debug('checking item %d at position %d', i, ipos)
        assert ipos == model_lim.items.get_loc(i)
        irates = mc_rates.loc[[i], :].set_index('user').rating

        ub_row = mat_ub.getrow(ipos)
        b_row = mat_lim.getrow(ipos)
        assert b_row.nnz <= MODEL_SIZE
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
            assert mat_ub[ipos, n] == approx(cor)

        # short rows are equal
        if b_row.nnz < MODEL_SIZE:
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
        assert len(b_nbrs) <= MODEL_SIZE
        assert all(b_nbrs.index.isin(ub_nbrs.index))
        # the similarities should be equal!
        b_match, ub_match = b_nbrs.align(ub_nbrs, join='inner')
        assert all(b_match == b_nbrs)
        assert b_match.values == approx(ub_match.values)
        assert b_nbrs.max() == approx(ub_nbrs.max())
        if len(ub_nbrs) > MODEL_SIZE:
            assert len(b_nbrs) == MODEL_SIZE
            ub_shrink = ub_nbrs.nlargest(MODEL_SIZE)
            # the minimums should be equal
            assert ub_shrink.min() == approx(b_nbrs.min())
            # everything above minimum value should be the same set of items
            ubs_except_min = ub_shrink[ub_shrink > b_nbrs.min()]
            assert all(ubs_except_min.index.isin(b_nbrs.index))


@mark.slow
def test_ii_save_load(tmp_path):
    "Save and load a model"
    tmp_path = lktu.norm_path(tmp_path)
    algo = knn.ItemItem(30, save_nbrs=500)
    _log.info('building model')
    original = algo.train(ml_ratings)

    fn = tmp_path / 'ii.mod'
    _log.info('saving model to %s', fn)
    algo.save_model(original, fn)
    _log.info('reloading model')
    model = algo.load_model(fn)
    _log.info('checking model')

    assert model is not None
    assert model is not original

    assert all(np.logical_not(np.isnan(model.sim_matrix.values)))
    assert all(model.sim_matrix.values > 0)
    # a little tolerance
    assert all(model.sim_matrix.values < 1 + 1.0e-6)

    assert all(model.counts == original.counts)
    assert model.counts.sum() == model.sim_matrix.nnz
    assert model.sim_matrix.nnz == original.sim_matrix.nnz
    assert all(model.sim_matrix.rowptrs == original.sim_matrix.rowptrs)
    assert model.sim_matrix.values == approx(original.sim_matrix.values)

    r_mat = model.sim_matrix
    o_mat = original.sim_matrix
    assert all(r_mat.rowptrs == o_mat.rowptrs)

    for i in range(len(model.items)):
        sp = r_mat.rowptrs[i]
        ep = r_mat.rowptrs[i + 1]

        # everything is in decreasing order
        assert all(np.diff(r_mat.values[sp:ep]) <= 0)
        assert all(r_mat.values[sp:ep] == o_mat.values[sp:ep])

    means = ml_ratings.groupby('item').rating.mean()
    assert means[model.items].values == approx(original.means)

    matrix = lm.csr_to_scipy(model.sim_matrix)

    items = pd.Series(model.items)
    items = items[model.counts > 0]
    for i in items.sample(50):
        ipos = model.items.get_loc(i)
        _log.debug('checking item %d at position %d', i, ipos)

        row = matrix.getrow(ipos)

        # it should be sorted !
        # check this by diffing the row values, and make sure they're negative
        assert all(np.diff(row.data) < 1.0e-6)


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_ii_batch_accuracy():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.load_ratings()

    uu_algo = knn.ItemItem(30)
    algo = basic.Fallback(uu_algo, basic.Bias())

    def eval(train, test):
        _log.info('running training')
        model = algo.train(train)
        _log.info('testing %d users', test.user.nunique())
        return batch.predict(lambda u, xs: algo.predict(model, u, xs), test)

    preds = pd.concat((eval(train, test)
                       for (train, test)
                       in xf.partition_users(ratings, 5, xf.SampleFrac(0.2))))
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.70, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.90, abs=0.05)


@mark.slow
def test_ii_known_preds():
    from lenskit import batch

    algo = knn.ItemItem(20, min_sim=1.0e-6)
    _log.info('training %s on ml data', algo)
    model = algo.train(lktu.ml_pandas.renamed.ratings)

    dir = Path(__file__).parent
    pred_file = dir / 'item-item-preds.csv'
    _log.info('reading known predictions from %s', pred_file)
    known_preds = pd.read_csv(str(pred_file))
    pairs = known_preds.loc[:, ['user', 'item']]

    preds = batch.predict(algo, pairs, model=model)
    merged = pd.merge(known_preds.rename(columns={'prediction': 'expected'}), preds)
    assert len(merged) == len(preds)
    merged['error'] = merged.expected - merged.prediction
    assert not any(merged.prediction.isna() & merged.expected.notna())
    err = merged.error
    err = err[err.notna()]
    try:
        assert all(err.abs() < 0.03)  # FIXME this threshold is too high
    except AssertionError as e:
        bad = merged[merged.error.notna() & (merged.error.abs() >= 0.01)]
        _log.error('erroneous predictions:\n%s', bad)
        raise e


@mark.slow
@mark.eval
def test_ii_batch_recommend():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch, topn
    import lenskit.metrics.topn as lm

    if not os.path.exists('ml-100k/u.data'):
        raise pytest.skip()

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    algo = knn.ItemItem(30)

    def eval(train, test):
        _log.info('running training')
        model = algo.train(train)
        _log.info('testing %d users', test.user.nunique())
        cand_fun = topn.UnratedCandidates(train)
        recs = batch.recommend(algo, model, test.user.unique(), 100, cand_fun)
        # combine with test ratings for relevance data
        res = pd.merge(recs, test, how='left', on=('user', 'item'))
        # fill in missing 0s
        res.loc[res.rating.isna(), 'rating'] = 0
        return res

    recs = pd.concat((eval(train, test)
                      for (train, test)
                      in xf.partition_users(ratings, 5, xf.SampleFrac(0.2))))

    _log.info('analyzing recommendations')
    ndcg = recs.groupby('user').rating.apply(lm.ndcg)
    _log.info('NDCG for %d users is %f', len(ndcg), ndcg.mean())
    assert ndcg.mean() > 0
