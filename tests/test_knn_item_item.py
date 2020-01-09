from lenskit import DataWarning
import lenskit.algorithms.item_knn as knn

from pathlib import Path
import logging
import os.path
import pickle

import pandas as pd
import numpy as np
from scipy import linalg as la

import pytest
from pytest import approx, mark

import lenskit.util.test as lktu

_log = logging.getLogger(__name__)

ml_ratings = lktu.ml_test.ratings
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
    algo.fit(simple_ratings)

    assert isinstance(algo.item_index_, pd.Index)
    assert isinstance(algo.item_means_, np.ndarray)
    assert isinstance(algo.item_counts_, np.ndarray)
    matrix = algo.sim_matrix_.to_scipy()

    # 6 is a neighbor of 7
    six, seven = algo.item_index_.get_indexer([6, 7])
    _log.info('six: %d', six)
    _log.info('seven: %d', seven)
    _log.info('matrix: %s', algo.sim_matrix_)
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

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values)))
    assert all(algo.sim_matrix_.values > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values < 1 + 1.0e-6)


def test_ii_train_unbounded():
    algo = knn.ItemItem(30)
    algo.fit(simple_ratings)

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values)))
    assert all(algo.sim_matrix_.values > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values < 1 + 1.0e-6)

    # 6 is a neighbor of 7
    matrix = algo.sim_matrix_.to_scipy()
    six, seven = algo.item_index_.get_indexer([6, 7])
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
    algo.fit(simple_ratings)

    res = algo.predict_for_user(3, [6])
    assert res is not None
    assert len(res) == 1
    assert 6 in res.index
    assert not np.isnan(res.loc[6])


def test_ii_simple_implicit_predict():
    algo = knn.ItemItem(30, center=False, aggregate='sum')
    algo.fit(simple_ratings.loc[:, ['user', 'item']])

    res = algo.predict_for_user(3, [6])
    assert res is not None
    assert len(res) == 1
    assert 6 in res.index
    assert not np.isnan(res.loc[6])
    assert res.loc[6] > 0


def test_ii_warn_duplicates():
    extra = pd.DataFrame.from_records([
        (3, 7, 4.5)
    ], columns=['user', 'item', 'rating'])
    ratings = pd.concat([simple_ratings, extra])
    algo = knn.ItemItem(5)
    algo.fit(ratings)

    try:
        with pytest.warns(DataWarning):
            algo.predict_for_user(3, [6])
    except AssertionError:
        pass  # this is fine


@lktu.wantjit
@mark.skip("redundant with large_models")
def test_ii_train_big():
    "Simple tests for bounded models"
    algo = knn.ItemItem(30, save_nbrs=500)
    algo.fit(ml_ratings)

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values)))
    assert all(algo.sim_matrix_.values > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values < 1 + 1.0e-6)

    assert algo.item_counts_.sum() == algo.sim_matrix_.nnz

    means = ml_ratings.groupby('item').rating.mean()
    assert means[algo.item_index_].values == approx(algo.item_means_)


@lktu.wantjit
@mark.skip("redundant with large_models")
def test_ii_train_big_unbounded():
    "Simple tests for unbounded models"
    algo = knn.ItemItem(30)
    algo.fit(ml_ratings)

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values)))
    assert all(algo.sim_matrix_.values > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values < 1 + 1.0e-6)

    assert algo.item_counts_.sum() == algo.sim_matrix_.nnz

    means = ml_ratings.groupby('item').rating.mean()
    assert means[algo.item_index_].values == approx(algo.item_means_)


@lktu.wantjit
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_ii_train_ml100k(tmp_path):
    "Test an unbounded model on ML-100K"
    ratings = lktu.ml100k.ratings
    algo = knn.ItemItem(30)
    _log.info('training model')
    algo.fit(ratings)

    _log.info('testing model')

    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values)))
    assert all(algo.sim_matrix_.values > 0)

    # a little tolerance
    assert all(algo.sim_matrix_.values < 1 + 1.0e-6)

    assert algo.item_counts_.sum() == algo.sim_matrix_.nnz

    means = ratings.groupby('item').rating.mean()
    assert means[algo.item_index_].values == approx(algo.item_means_)

    # save
    fn = tmp_path / 'ii.mod'
    _log.info('saving model to %s', fn)
    with fn.open('wb') as modf:
        pickle.dump(algo, modf)

    _log.info('reloading model')
    with fn.open('rb') as modf:
        restored = pickle.load(modf)

    assert all(restored.sim_matrix_.values > 0)

    r_mat = restored.sim_matrix_
    o_mat = algo.sim_matrix_

    assert all(r_mat.rowptrs == o_mat.rowptrs)

    for i in range(len(restored.item_index_)):
        sp = r_mat.rowptrs[i]
        ep = r_mat.rowptrs[i + 1]

        # everything is in decreasing order
        assert all(np.diff(r_mat.values[sp:ep]) <= 0)
        assert all(r_mat.values[sp:ep] == o_mat.values[sp:ep])


@lktu.wantjit
@mark.slow
def test_ii_large_models():
    "Several tests of large trained I-I models"
    _log.info('training limited model')
    MODEL_SIZE = 100
    algo_lim = knn.ItemItem(30, save_nbrs=MODEL_SIZE)
    algo_lim.fit(ml_ratings)

    _log.info('training unbounded model')
    algo_ub = knn.ItemItem(30)
    algo_ub.fit(ml_ratings)

    _log.info('testing models')
    assert all(np.logical_not(np.isnan(algo_lim.sim_matrix_.values)))
    assert all(algo_lim.sim_matrix_.values > 0)
    # a little tolerance
    assert all(algo_lim.sim_matrix_.values < 1 + 1.0e-6)

    means = ml_ratings.groupby('item').rating.mean()
    assert means[algo_lim.item_index_].values == approx(algo_lim.item_means_)

    assert all(np.logical_not(np.isnan(algo_ub.sim_matrix_.values)))
    assert all(algo_ub.sim_matrix_.values > 0)
    # a little tolerance
    assert all(algo_ub.sim_matrix_.values < 1 + 1.0e-6)

    means = ml_ratings.groupby('item').rating.mean()
    assert means[algo_ub.item_index_].values == approx(algo_ub.item_means_)

    mc_rates = ml_ratings.set_index('item')\
                         .join(pd.DataFrame({'item_mean': means}))\
                         .assign(rating=lambda df: df.rating - df.item_mean)

    mat_lim = algo_lim.sim_matrix_.to_scipy()
    mat_ub = algo_ub.sim_matrix_.to_scipy()

    _log.info('checking a sample of neighborhoods')
    items = pd.Series(algo_ub.item_index_)
    items = items[algo_ub.item_counts_ > 0]
    for i in items.sample(50):
        ipos = algo_ub.item_index_.get_loc(i)
        _log.debug('checking item %d at position %d', i, ipos)
        assert ipos == algo_lim.item_index_.get_loc(i)
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
            n_id = algo_ub.item_index_[n]
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
        ub_nbrs = pd.Series(ub_row.data, algo_ub.item_index_[ub_row.indices])
        b_nbrs = pd.Series(b_row.data, algo_lim.item_index_[b_row.indices])

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


@lktu.wantjit
def test_ii_save_load(tmp_path):
    "Save and load a model"
    original = knn.ItemItem(30, save_nbrs=500)
    _log.info('building model')
    original.fit(lktu.ml_sample())

    fn = tmp_path / 'ii.mod'
    _log.info('saving model to %s', fn)
    with fn.open('wb') as modf:
        pickle.dump(original, modf)

    _log.info('reloading model')
    with fn.open('rb') as modf:
        algo = pickle.load(modf)

    _log.info('checking model')
    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values)))
    assert all(algo.sim_matrix_.values > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values < 1 + 1.0e-6)

    assert all(algo.item_counts_ == original.item_counts_)
    assert algo.item_counts_.sum() == algo.sim_matrix_.nnz
    assert algo.sim_matrix_.nnz == original.sim_matrix_.nnz
    assert all(algo.sim_matrix_.rowptrs == original.sim_matrix_.rowptrs)
    assert algo.sim_matrix_.values == approx(original.sim_matrix_.values)

    r_mat = algo.sim_matrix_
    o_mat = original.sim_matrix_
    assert all(r_mat.rowptrs == o_mat.rowptrs)

    for i in range(len(algo.item_index_)):
        sp = r_mat.rowptrs[i]
        ep = r_mat.rowptrs[i + 1]

        # everything is in decreasing order
        assert all(np.diff(r_mat.values[sp:ep]) <= 0)
        assert all(r_mat.values[sp:ep] == o_mat.values[sp:ep])

    means = ml_ratings.groupby('item').rating.mean()
    assert means[algo.item_index_].values == approx(original.item_means_)

    matrix = algo.sim_matrix_.to_scipy()

    items = pd.Series(algo.item_index_)
    items = items[algo.item_counts_ > 0]
    for i in items.sample(50):
        ipos = algo.item_index_.get_loc(i)
        _log.debug('checking item %d at position %d', i, ipos)

        row = matrix.getrow(ipos)

        # it should be sorted !
        # check this by diffing the row values, and make sure they're negative
        assert all(np.diff(row.data) < 1.0e-6)


def test_ii_implicit_save_load(tmp_path):
    "Save and load a model"
    original = knn.ItemItem(30, save_nbrs=500, center=False, aggregate='sum')
    _log.info('building model')
    original.fit(lktu.ml_sample().loc[:, ['user', 'item']])

    fn = tmp_path / 'ii.mod'
    _log.info('saving model to %s', fn)
    with fn.open('wb') as modf:
        pickle.dump(original, modf)

    _log.info('reloading model')
    with fn.open('rb') as modf:
        algo = pickle.load(modf)

    _log.info('checking model')
    assert all(np.logical_not(np.isnan(algo.sim_matrix_.values)))
    assert all(algo.sim_matrix_.values > 0)
    # a little tolerance
    assert all(algo.sim_matrix_.values < 1 + 1.0e-6)

    assert all(algo.item_counts_ == original.item_counts_)
    assert algo.item_counts_.sum() == algo.sim_matrix_.nnz
    assert algo.sim_matrix_.nnz == original.sim_matrix_.nnz
    assert all(algo.sim_matrix_.rowptrs == original.sim_matrix_.rowptrs)
    assert algo.sim_matrix_.values == approx(original.sim_matrix_.values)
    assert algo.rating_matrix_.values is None

    r_mat = algo.sim_matrix_
    o_mat = original.sim_matrix_
    assert all(r_mat.rowptrs == o_mat.rowptrs)

    for i in range(len(algo.item_index_)):
        sp = r_mat.rowptrs[i]
        ep = r_mat.rowptrs[i + 1]

        # everything is in decreasing order
        assert all(np.diff(r_mat.values[sp:ep]) <= 0)
        assert all(r_mat.values[sp:ep] == o_mat.values[sp:ep])

    assert algo.item_means_ is None

    matrix = algo.sim_matrix_.to_scipy()

    items = pd.Series(algo.item_index_)
    items = items[algo.item_counts_ > 0]
    for i in items.sample(50):
        ipos = algo.item_index_.get_loc(i)
        _log.debug('checking item %d at position %d', i, ipos)

        row = matrix.getrow(ipos)

        # it should be sorted !
        # check this by diffing the row values, and make sure they're negative
        assert all(np.diff(row.data) < 1.0e-6)


@lktu.wantjit
@mark.slow
def test_ii_implicit():
    algo = knn.ItemItem(20, save_nbrs=100, center=False, aggregate='sum')
    data = ml_ratings.loc[:, ['user', 'item']]

    algo.fit(data)
    assert algo.item_counts_.sum() == algo.sim_matrix_.nnz
    assert all(algo.sim_matrix_.values > 0)
    assert all(algo.item_counts_ <= 100)

    preds = algo.predict_for_user(50, [1, 2, 42])
    assert all(preds[preds.notna()] > 0)


@mark.slow
def test_ii_implicit_fast_ident():
    algo = knn.ItemItem(20, save_nbrs=100, center=False, aggregate='sum')
    data = ml_ratings.loc[:, ['user', 'item']]

    algo.fit(data)
    assert algo.item_counts_.sum() == algo.sim_matrix_.nnz
    assert all(algo.sim_matrix_.values > 0)
    assert all(algo.item_counts_ <= 100)

    preds = algo.predict_for_user(50, [1, 2, 42])
    assert all(preds[preds.notna()] > 0)
    assert np.isnan(preds.iloc[2])

    algo.min_sim = -1  # force it to take the slow path for all predictions
    p2 = algo.predict_for_user(50, [1, 2, 42])
    assert preds.values[:2] == approx(p2.values[:2])
    assert np.isnan(p2.iloc[2])


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_ii_batch_accuracy():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.ratings

    ii_algo = knn.ItemItem(30)
    algo = basic.Fallback(ii_algo, basic.Bias())

    def eval(train, test):
        _log.info('running training')
        algo.fit(train)
        _log.info('testing %d users', test.user.nunique())
        return batch.predict(algo, test, nprocs=4)

    preds = pd.concat((eval(train, test)
                       for (train, test)
                       in xf.partition_users(ratings, 5, xf.SampleFrac(0.2))))
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.70, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.90, abs=0.05)


@lktu.wantjit
@mark.slow
def test_ii_known_preds():
    from lenskit import batch

    algo = knn.ItemItem(20, min_sim=1.0e-6)
    _log.info('training %s on ml data', algo)
    algo.fit(lktu.ml_test.ratings)
    assert algo.center
    assert algo.item_means_ is not None
    _log.info('model means: %s', algo.item_means_)

    dir = Path(__file__).parent
    pred_file = dir / 'item-item-preds.csv'
    _log.info('reading known predictions from %s', pred_file)
    known_preds = pd.read_csv(str(pred_file))
    pairs = known_preds.loc[:, ['user', 'item']]

    preds = batch.predict(algo, pairs)
    merged = pd.merge(known_preds.rename(columns={'prediction': 'expected'}), preds)
    assert len(merged) == len(preds)
    merged['error'] = merged.expected - merged.prediction
    try:
        assert not any(merged.prediction.isna() & merged.expected.notna())
    except AssertionError as e:
        bad = merged[merged.prediction.isna() & merged.expected.notna()]
        _log.error('erroneously missing or present predictions:\n%s', bad)
        raise e

    err = merged.error
    err = err[err.notna()]
    try:
        assert all(err.abs() < 0.03)  # FIXME this threshold is too high
    except AssertionError as e:
        bad = merged[merged.error.notna() & (merged.error.abs() >= 0.01)]
        _log.error('erroneous predictions:\n%s', bad)
        raise e


@lktu.wantjit
@mark.skipif(knn._mkl_ops is None, reason='only test MKL match when MKL is available')
def test_ii_impl_match():
    from lenskit import batch

    sps = knn.ItemItem(20, min_sim=1.0e-6)
    sps._use_mkl = False
    _log.info('training SciPy %s on ml data', sps)
    sps.fit(lktu.ml_test.ratings)

    mkl = knn.ItemItem(20, min_sim=1.0e-6)
    _log.info('training MKL %s on ml data', mkl)
    mkl.fit(lktu.ml_test.ratings)

    assert mkl.sim_matrix_.nnz == sps.sim_matrix_.nnz
    assert mkl.sim_matrix_.nrows == sps.sim_matrix_.nrows
    assert mkl.sim_matrix_.ncols == sps.sim_matrix_.ncols

    assert all(mkl.sim_matrix_.rowptrs == sps.sim_matrix_.rowptrs)
    for i in range(mkl.sim_matrix_.nrows):
        sp, ep = mkl.sim_matrix_.row_extent(i)
        assert all(np.diff(mkl.sim_matrix_.values[sp:ep]) <= 0)
        assert all(np.diff(sps.sim_matrix_.values[sp:ep]) <= 0)
        assert set(mkl.sim_matrix_.colinds[sp:ep]) == set(sps.sim_matrix_.colinds[sp:ep])


@lktu.wantjit
@mark.slow
@mark.eval
@mark.parametrize('ncpus', [1, 2])
def test_ii_batch_recommend(ncpus):
    import lenskit.crossfold as xf
    from lenskit import batch, topn

    if not os.path.exists('ml-100k/u.data'):
        raise pytest.skip()

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    algo = knn.ItemItem(30)

    def eval(train, test):
        _log.info('running training')
        algo.fit(train)
        _log.info('testing %d users', test.user.nunique())
        cand_fun = topn.UnratedCandidates(train)
        recs = batch.recommend(algo, test.user.unique(), 100, cand_fun, n_jobs=ncpus)
        return recs

    test_frames = []
    recs = []
    for train, test in xf.partition_users(ratings, 5, xf.SampleFrac(0.2)):
        test_frames.append(test)
        recs.append(eval(train, test))

    test = pd.concat(test_frames)
    recs = pd.concat(recs)

    _log.info('analyzing recommendations')
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, test)
    dcg = results.ndcg
    _log.info('nDCG for %d users is %f', len(dcg), dcg.mean())
    assert dcg.mean() > 0.03
