from lenskit import matrix
import lenskit.algorithms.user_knn as knn

from pathlib import Path
import logging

import pandas as pd
import numpy as np
from scipy import sparse as sps

from pytest import approx, mark

import lk_test_utils as lktu

_log = logging.getLogger(__name__)

ml_ratings = lktu.ml_pandas.renamed.ratings


def test_uu_train():
    algo = knn.UserUser(30)
    model = algo.train(ml_ratings)

    # it should have computed correct means
    umeans = ml_ratings.groupby('user').rating.mean()
    mlmeans = pd.Series(model.user_means, index=model.users, name='mean')
    umeans, mlmeans = umeans.align(mlmeans)
    assert mlmeans.values == approx(umeans.values)

    # we should be able to reconstruct rating values
    uir = ml_ratings.set_index(['user', 'item']).rating
    r_items = matrix.csr_rowinds(model.transpose)
    ui_rbdf = pd.DataFrame({
        'user': model.users[model.transpose.colinds],
        'item': model.items[r_items],
        'nrating': model.transpose.values
    }).set_index(['user', 'item'])
    ui_rbdf = ui_rbdf.join(mlmeans)
    ui_rbdf['rating'] = ui_rbdf['nrating'] + ui_rbdf['mean']
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
    mlmeans = pd.Series(model.user_means, index=model.users)
    umeans, mlmeans = umeans.align(mlmeans)
    assert mlmeans.values == approx(umeans.values)

    # we should be able to reconstruct rating values
    uir = ml_ratings.set_index(['user', 'item']).rating
    r_items = matrix.csr_rowinds(model.transpose)
    ui_rbdf = pd.DataFrame({
        'user': model.user_means.index[model.transpose.colinds],
        'item': model.items[r_items],
        'nrating': model.transpose.values
    }).set_index(['user', 'item'])
    ui_rbdf = ui_rbdf.join(model.user_means)
    ui_rbdf['rating'] = ui_rbdf['nrating'] + ui_rbdf['mean']
    ui_rbdf['orig_rating'] = uir
    assert ui_rbdf.rating.values == approx(ui_rbdf.orig_rating.values)


def test_uu_predict_unknown_empty():
    algo = knn.UserUser(30, min_nbrs=2)
    model = algo.train(ml_ratings)

    preds = algo.predict(model, -28018, [1016, 2091])
    assert len(preds) == 2
    assert all(preds.isna())


def test_uu_implicit():
    "Train and use user-user on an implicit data set."
    algo = knn.UserUser(20, center=False, aggregate='sum')
    data = ml_ratings.loc[:, ['user', 'item']]

    model = algo.train(data)
    assert model is not None
    assert model.user_means is None

    mat = matrix.csr_to_scipy(model.matrix)
    norms = sps.linalg.norm(mat, 2, 1)
    assert norms == approx(1.0)

    preds = algo.predict(model, 50, [1, 2, 42])
    assert all(preds[preds.notna()] > 0)


@mark.slow
def test_uu_save_load_implicit(tmp_path):
    "Save and load user-user on an implicit data set."
    algo = knn.UserUser(20, center=False, aggregate='sum')
    data = ml_ratings.loc[:, ['user', 'item']]

    original = algo.train(data)
    algo.save_model(original, tmp_path / 'uu.mod')

    model = algo.load_model(tmp_path / 'uu.mod')
    assert model is not None
    assert model.user_means is None
    assert all(model.users == original.users)
    assert all(model.items == original.items)

    assert all(model.matrix.rowptrs == original.matrix.rowptrs)
    assert all(model.matrix.colinds == original.matrix.colinds)
    assert all(model.matrix.values == original.matrix.values)

    assert all(model.transpose.rowptrs == original.transpose.rowptrs)
    assert all(model.transpose.colinds == original.transpose.colinds)
    assert model.transpose.values is None


@mark.slow
def test_uu_known_preds():
    from lenskit import batch

    algo = knn.UserUser(30, min_sim=1.0e-6)
    _log.info('training %s on ml data', algo)
    model = algo.train(lktu.ml_pandas.renamed.ratings)

    dir = Path(__file__).parent
    pred_file = dir / 'user-user-preds.csv'
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
        assert all(err.abs() < 0.01)
    except AssertionError as e:
        bad = merged[merged.error.notna() & (merged.error.abs() >= 0.01)]
        _log.error('erroneous predictions:\n%s', bad)
        raise e


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
    preds = [__batch_eval((algo, train, test)) for (train, test) in folds]
    preds = pd.concat(preds)
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.71, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.91, abs=0.05)


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_uu_implicit_batch_accuracy():
    from lenskit import batch, topn
    import lenskit.crossfold as xf
    import lenskit.metrics.topn as lm

    ratings = lktu.ml100k.load_ratings()

    algo = knn.UserUser(30, center=False, aggregate='sum')

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    rec_lists = []
    for train, test in folds:
        _log.info('running training')
        model = algo.train(train.loc[:, ['user', 'item']])
        cands = topn.UnratedCandidates(train)
        _log.info('testing %d users', test.user.nunique())
        recs = batch.recommend(algo, model, test.user.unique(), 100, cands, test)
        rec_lists.append(recs)
    recs = pd.concat(rec_lists)

    user_ndcg = recs.groupby('user').rating.apply(lm.ndcg)
    ndcg = user_ndcg.mean()
    assert ndcg >= 0.1
