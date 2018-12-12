import pytest

import sys
import os
import os.path
from collections import namedtuple
from functools import partial
import logging
import warnings
import pandas as pd
import numpy as np

import lk_test_utils as lktu

from lenskit.algorithms.basic import Bias
import lenskit.batch as lkb

MLB = namedtuple('MLB', ['ratings', 'algo', 'model'])
MLB.predictor = property(lambda mlb: partial(mlb.algo.predict, mlb.model))
_log = logging.getLogger(__name__)


@pytest.fixture
def mlb():
    ratings = lktu.ml_pandas.renamed.ratings
    algo = Bias()
    model = algo.train(ratings)
    return MLB(ratings, algo, model)


def test_recommend_single(mlb):
    res = lkb.recommend(mlb.algo, mlb.model, [1], None, {1: [31]})

    assert len(res) == 1
    assert all(res['user'] == 1)
    assert all(res['rank'] == 1)
    assert set(res.columns) == set(['user', 'rank', 'item', 'score'])

    expected = mlb.model.mean + mlb.model.items.loc[31] + mlb.model.users.loc[1]
    assert res.score.iloc[0] == pytest.approx(expected)


def test_recommend_user(mlb):
    uid = 5
    items = mlb.ratings.item.unique()

    def candidates(user):
        urs = mlb.ratings[mlb.ratings.user == user]
        return np.setdiff1d(items, urs.item.unique())

    res = lkb.recommend(mlb.algo, mlb.model, [5], 10, candidates)

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

    res = lkb.recommend(mlb.algo, mlb.model, [5, 10], 10, candidates)

    assert len(res) == 20
    assert set(res.user) == set([5, 10])
    assert all(res.groupby('user').item.count() == 10)
    assert all(res.groupby('user')['rank'].max() == 10)
    assert all(np.diff(res[res.user == 5].score) <= 0)
    assert all(np.diff(res[res.user == 5]['rank']) == 1)
    assert all(np.diff(res[res.user == 10].score) <= 0)
    assert all(np.diff(res[res.user == 10]['rank']) == 1)


def test_bias_batch_recommend():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch, topn
    import lenskit.metrics.topn as lm

    if not os.path.exists('ml-100k/u.data'):
        raise pytest.skip()

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    algo = basic.Bias(damping=5)

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
    dcg = recs.groupby('user').rating.apply(lm.dcg)
    _log.info('DCG for %d users is %f (max=%f)', len(dcg), dcg.mean(), dcg.max())
    assert dcg.mean() > 0


@pytest.mark.parametrize('ncpus', [None, 2])
def test_pop_batch_recommend(ncpus):
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch, topn
    import lenskit.metrics.topn as lm

    if not os.path.exists('ml-100k/u.data'):
        raise pytest.skip()

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    algo = basic.Popular()

    def eval(train, test):
        _log.info('running training')
        model = algo.train(train)
        _log.info('testing %d users', test.user.nunique())
        cand_fun = topn.UnratedCandidates(train)
        recs = batch.recommend(algo, model, test.user.unique(), 100, cand_fun,
                               test, nprocs=ncpus)
        return recs

    recs = pd.concat((eval(train, test)
                      for (train, test)
                      in xf.partition_users(ratings, 5, xf.SampleFrac(0.2))))

    _log.info('analyzing recommendations')
    _log.info('have %d recs for good items', (recs.rating > 0).sum())
    dcg = recs.groupby('user').rating.agg(lm.dcg)
    _log.info('DCG for %d users is %f (max=%f)', len(dcg), dcg.mean(), dcg.max())
    assert dcg.mean() > 0
