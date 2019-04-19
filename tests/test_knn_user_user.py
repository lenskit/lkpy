import lenskit.algorithms.user_knn as knn

from pathlib import Path
import logging
import pickle

import pandas as pd
import numpy as np
from scipy import sparse as sps

from pytest import approx, mark

import lenskit.util.test as lktu

_log = logging.getLogger(__name__)

ml_ratings = lktu.ml_test.ratings


def test_uu_train():
    algo = knn.UserUser(30)
    ret = algo.fit(ml_ratings)
    assert ret is algo

    # it should have computed correct means
    umeans = ml_ratings.groupby('user').rating.mean()
    mlmeans = pd.Series(algo.user_means_, index=algo.user_index_, name='mean')
    umeans, mlmeans = umeans.align(mlmeans)
    assert mlmeans.values == approx(umeans.values)

    # we should be able to reconstruct rating values
    uir = ml_ratings.set_index(['user', 'item']).rating
    r_items = algo.transpose_matrix_.rowinds()
    ui_rbdf = pd.DataFrame({
        'user': algo.user_index_[algo.transpose_matrix_.colinds],
        'item': algo.item_index_[r_items],
        'nrating': algo.transpose_matrix_.values
    }).set_index(['user', 'item'])
    ui_rbdf = ui_rbdf.join(mlmeans)
    ui_rbdf['rating'] = ui_rbdf['nrating'] + ui_rbdf['mean']
    ui_rbdf['orig_rating'] = uir
    assert ui_rbdf.rating.values == approx(ui_rbdf.orig_rating.values)


def test_uu_predict_one():
    algo = knn.UserUser(30)
    algo.fit(ml_ratings)

    preds = algo.predict_for_user(4, [1016])
    assert len(preds) == 1
    assert preds.index == [1016]
    assert preds.values == approx([3.62221550680778])


def test_uu_predict_too_few():
    algo = knn.UserUser(30, min_nbrs=2)
    algo.fit(ml_ratings)

    preds = algo.predict_for_user(4, [2091])
    assert len(preds) == 1
    assert preds.index == [2091]
    assert all(preds.isna())


def test_uu_predict_too_few_blended():
    algo = knn.UserUser(30, min_nbrs=2)
    algo.fit(ml_ratings)

    preds = algo.predict_for_user(4, [1016, 2091])
    assert len(preds) == 2
    assert np.isnan(preds.loc[2091])
    assert preds.loc[1016] == approx(3.62221550680778)


def test_uu_predict_live_ratings():
    algo = knn.UserUser(30, min_nbrs=2)
    no4 = ml_ratings[ml_ratings.user != 4]
    algo.fit(no4)

    ratings = ml_ratings[ml_ratings.user == 4].set_index('item').rating

    preds = algo.predict_for_user(20381, [1016, 2091], ratings)
    assert len(preds) == 2
    assert np.isnan(preds.loc[2091])
    assert preds.loc[1016] == approx(3.62221550680778)


def test_uu_save_load(tmp_path):
    orig = knn.UserUser(30)
    _log.info('training model')
    orig.fit(ml_ratings)

    fn = tmp_path / 'uu.model'
    _log.info('saving to %s', fn)
    with fn.open('wb') as f:
        pickle.dump(orig, f)

    _log.info('reloading model')
    with fn.open('rb') as f:
        algo = pickle.load(f)

    _log.info('checking model')

    # it should have computed correct means
    umeans = ml_ratings.groupby('user').rating.mean()
    mlmeans = pd.Series(algo.user_means_, index=algo.user_index_, name='mean')
    umeans, mlmeans = umeans.align(mlmeans)
    assert mlmeans.values == approx(umeans.values)

    # we should be able to reconstruct rating values
    uir = ml_ratings.set_index(['user', 'item']).rating
    r_items = algo.transpose_matrix_.rowinds()
    ui_rbdf = pd.DataFrame({
        'user': algo.user_index_[algo.transpose_matrix_.colinds],
        'item': algo.item_index_[r_items],
        'nrating': algo.transpose_matrix_.values
    }).set_index(['user', 'item'])
    ui_rbdf = ui_rbdf.join(mlmeans)
    ui_rbdf['rating'] = ui_rbdf['nrating'] + ui_rbdf['mean']
    ui_rbdf['orig_rating'] = uir
    assert ui_rbdf.rating.values == approx(ui_rbdf.orig_rating.values)

    # running the predictor should work
    preds = algo.predict_for_user(4, [1016])
    assert len(preds) == 1
    assert preds.index == [1016]
    assert preds.values == approx([3.62221550680778])


def test_uu_predict_unknown_empty():
    algo = knn.UserUser(30, min_nbrs=2)
    algo.fit(ml_ratings)

    preds = algo.predict_for_user(-28018, [1016, 2091])
    assert len(preds) == 2
    assert all(preds.isna())


def test_uu_implicit():
    "Train and use user-user on an implicit data set."
    algo = knn.UserUser(20, center=False, aggregate='sum')
    data = ml_ratings.loc[:, ['user', 'item']]

    algo.fit(data)
    assert algo.user_means_ is None

    mat = algo.rating_matrix_.to_scipy()
    norms = sps.linalg.norm(mat, 2, 1)
    assert norms == approx(1.0)

    preds = algo.predict_for_user(50, [1, 2, 42])
    assert all(preds[preds.notna()] > 0)


@mark.slow
def test_uu_save_load_implicit(tmp_path):
    "Save and load user-user on an implicit data set."
    orig = knn.UserUser(20, center=False, aggregate='sum')
    data = ml_ratings.loc[:, ['user', 'item']]

    orig.fit(data)
    ser = pickle.dumps(orig)

    algo = pickle.loads(ser)

    assert algo.user_means_ is None
    assert all(algo.user_index_ == orig.user_index_)
    assert all(algo.item_index_ == orig.item_index_)

    assert all(algo.rating_matrix_.rowptrs == orig.rating_matrix_.rowptrs)
    assert all(algo.rating_matrix_.colinds == orig.rating_matrix_.colinds)
    assert all(algo.rating_matrix_.values == orig.rating_matrix_.values)

    assert all(algo.transpose_matrix_.rowptrs == orig.transpose_matrix_.rowptrs)
    assert all(algo.transpose_matrix_.colinds == orig.transpose_matrix_.colinds)
    assert algo.transpose_matrix_.values is None


@mark.slow
def test_uu_known_preds():
    from lenskit import batch

    algo = knn.UserUser(30, min_sim=1.0e-6)
    _log.info('training %s on ml data', algo)
    algo.fit(lktu.ml_test.ratings)

    dir = Path(__file__).parent
    pred_file = dir / 'user-user-preds.csv'
    _log.info('reading known predictions from %s', pred_file)
    known_preds = pd.read_csv(str(pred_file))
    pairs = known_preds.loc[:, ['user', 'item']]

    preds = batch.predict(algo, pairs)
    merged = pd.merge(known_preds.rename(columns={'prediction': 'expected'}), preds)
    assert len(merged) == len(preds)
    merged['error'] = merged.expected - merged.prediction
    assert not any(merged.prediction.isna() & merged.expected.notna())
    err = merged.error
    err = err[err.notna()]
    try:
        assert all(err.abs() < 0.01)
    except AssertionError as e:
        bad = merged[merged.error.notna() & (merged.error.abs() >= 0.01)]
        _log.error('erroneous predictions:\n%s', bad)
        raise e


def __batch_eval(job):
    from lenskit import batch
    algo, train, test = job
    _log.info('running training')
    algo.fit(train)
    _log.info('testing %d users', test.user.nunique())
    return batch.predict(algo, test)


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_uu_batch_accuracy():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.ratings

    uu_algo = knn.UserUser(30)
    algo = basic.Fallback(uu_algo, basic.Bias())

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    preds = [__batch_eval((algo, train, test)) for (train, test) in folds]
    preds = pd.concat(preds)
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.71, abs=0.028)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.91, abs=0.055)


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_uu_implicit_batch_accuracy():
    from lenskit import batch, topn
    import lenskit.crossfold as xf

    ratings = lktu.ml100k.ratings

    algo = knn.UserUser(30, center=False, aggregate='sum')

    folds = list(xf.partition_users(ratings, 5, xf.SampleFrac(0.2)))
    all_test = pd.concat(f.test for f in folds)

    rec_lists = []
    for train, test in folds:
        _log.info('running training')
        algo.fit(train.loc[:, ['user', 'item']])
        cands = topn.UnratedCandidates(train)
        _log.info('testing %d users', test.user.nunique())
        recs = batch.recommend(algo, test.user.unique(), 100, cands, n_jobs=2)
        rec_lists.append(recs)
    recs = pd.concat(rec_lists)

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, all_test)
    user_dcg = results.ndcg

    dcg = user_dcg.mean()
    assert dcg >= 0.03
