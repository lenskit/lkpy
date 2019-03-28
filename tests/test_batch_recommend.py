import pytest

import os
import os.path
from collections import namedtuple
import logging
import pandas as pd
import numpy as np

import lk_test_utils as lktu

from lenskit.algorithms.basic import Bias, TopN
import lenskit.batch as lkb

MLB = namedtuple('MLB', ['ratings', 'algo'])
_log = logging.getLogger(__name__)


@pytest.fixture
def mlb():
    ratings = lktu.ml_pandas.renamed.ratings
    algo = TopN(Bias())
    algo.fit(ratings)
    return MLB(ratings, algo)


def test_recommend_single(mlb):
    res = lkb.recommend(mlb.algo, [1], None, {1: [31]})

    assert len(res) == 1
    assert all(res['user'] == 1)
    assert all(res['rank'] == 1)
    assert set(res.columns) == set(['user', 'rank', 'item', 'score'])

    algo = mlb.algo.predictor
    expected = algo.mean_ + algo.item_offsets_.loc[31] + algo.user_offsets_.loc[1]
    assert res.score.iloc[0] == pytest.approx(expected)


def test_recommend_user(mlb):
    uid = 5
    items = mlb.ratings.item.unique()

    def candidates(user):
        urs = mlb.ratings[mlb.ratings.user == user]
        return np.setdiff1d(items, urs.item.unique())

    res = lkb.recommend(mlb.algo, [5], 10, candidates)

    assert len(res) == 10
    assert set(res.columns) == set(['user', 'rank', 'item', 'score'])
    assert all(res['user'] == uid)
    assert all(res['rank'] == np.arange(10) + 1)
    # they should be in decreasing order
    assert all(np.diff(res.score) <= 0)


def test_recommend_two_users(mlb):
    items = mlb.ratings.item.unique()

    def candidates(user):
        urs = mlb.ratings[mlb.ratings.user == user]
        return np.setdiff1d(items, urs.item.unique())

    res = lkb.recommend(mlb.algo, [5, 10], 10, candidates)

    assert len(res) == 20
    assert set(res.user) == set([5, 10])
    assert all(res.groupby('user').item.count() == 10)
    assert all(res.groupby('user')['rank'].max() == 10)
    assert all(np.diff(res[res.user == 5].score) <= 0)
    assert all(np.diff(res[res.user == 5]['rank']) == 1)
    assert all(np.diff(res[res.user == 10].score) <= 0)
    assert all(np.diff(res[res.user == 10]['rank']) == 1)


def test_recommend_no_cands(mlb):
    res = lkb.recommend(mlb.algo, [5, 10], 10)

    assert len(res) == 20
    assert set(res.user) == set([5, 10])
    assert all(res.groupby('user').item.count() == 10)
    assert all(res.groupby('user')['rank'].max() == 10)
    assert all(np.diff(res[res.user == 5].score) <= 0)
    assert all(np.diff(res[res.user == 5]['rank']) == 1)
    assert all(np.diff(res[res.user == 10].score) <= 0)
    assert all(np.diff(res[res.user == 10]['rank']) == 1)

    idx_rates = mlb.ratings.set_index(['user', 'item'])
    merged = res.join(idx_rates, on=['user', 'item'], how='inner')
    assert len(merged) == 0


@pytest.mark.eval
def test_bias_batch_recommend():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch, topn

    if not os.path.exists('ml-100k/u.data'):
        raise pytest.skip()

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    algo = basic.Bias(damping=5)
    algo = TopN(algo)

    def eval(train, test):
        _log.info('running training')
        algo.fit(train)
        _log.info('testing %d users', test.user.nunique())
        recs = batch.recommend(algo, test.user.unique(), 100)
        return recs

    folds = list(xf.partition_users(ratings, 5, xf.SampleFrac(0.2)))
    test = pd.concat(y for (x, y) in folds)

    recs = pd.concat(eval(train, test) for (train, test) in folds)

    _log.info('analyzing recommendations')
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, test)
    dcg = results.ndcg
    _log.info('nDCG for %d users is %f (max=%f)', len(dcg), dcg.mean(), dcg.max())
    assert dcg.mean() > 0


@pytest.mark.parametrize('ncpus', [None, 2])
@pytest.mark.eval
def test_pop_batch_recommend(ncpus):
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch, topn

    if not os.path.exists('ml-100k/u.data'):
        raise pytest.skip()

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    algo = basic.Popular()

    def eval(train, test):
        _log.info('running training')
        algo.fit(train)
        _log.info('testing %d users', test.user.nunique())
        recs = batch.recommend(algo, test.user.unique(), 100,
                               nprocs=ncpus)
        return recs

    folds = list(xf.partition_users(ratings, 5, xf.SampleFrac(0.2)))
    test = pd.concat(f.test for f in folds)

    recs = pd.concat(eval(train, test) for (train, test) in folds)

    _log.info('analyzing recommendations')
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, test)
    dcg = results.ndcg
    _log.info('NDCG for %d users is %f (max=%f)', len(dcg), dcg.mean(), dcg.max())
    assert dcg.mean() > 0
