import pytest

import logging
from collections import namedtuple
import pandas as pd
import numpy as np

import lenskit.util.test as lktu

from lenskit.algorithms.basic import Bias
import lenskit.batch as lkb

_log = logging.getLogger(__name__)

MLB = namedtuple('MLB', ['ratings', 'algo'])


@pytest.fixture
def mlb():
    ratings = lktu.ml_test.ratings
    algo = Bias()
    algo.fit(ratings)
    return MLB(ratings, algo)


def test_predict_single(mlb):
    tf = pd.DataFrame({'user': [1], 'item': [31]})
    res = lkb.predict(mlb.algo, tf)

    assert len(res) == 1
    assert all(res.user == 1)
    assert set(res.columns) == set(['user', 'item', 'prediction'])
    assert all(res.item == 31)

    expected = mlb.algo.mean_ + mlb.algo.item_offsets_.loc[31] + mlb.algo.user_offsets_.loc[1]
    assert res.prediction.iloc[0] == pytest.approx(expected)


def test_predict_user(mlb):
    uid = 5
    urates = mlb.ratings[mlb.ratings.user == uid]

    test_rated = urates.item.sample(5)
    unrated = np.setdiff1d(mlb.ratings.item.unique(), urates.item.values)
    test_unrated = np.random.choice(unrated, 10, replace=False)
    test_items = pd.concat([test_rated, pd.Series(test_unrated)])

    tf = pd.DataFrame({'user': uid, 'item': test_items})
    res = lkb.predict(mlb.algo, tf)

    assert len(res) == 15
    assert set(res.columns) == set(['user', 'item', 'prediction'])
    assert all(res.user == uid)
    assert set(res.item) == set(test_items)

    # did we get the right predictions?
    preds = res.set_index(['user', 'item'])
    preds['rating'] = mlb.algo.mean_
    preds['rating'] += mlb.algo.item_offsets_
    preds['rating'] += mlb.algo.user_offsets_.loc[uid]
    assert preds.prediction.values == pytest.approx(preds.rating.values)


def test_predict_two_users(mlb):
    uids = [5, 10]
    tf = None
    # make sure we get both UIDs
    while tf is None or len(set(tf.user)) < 2:
        tf = mlb.ratings[mlb.ratings.user.isin(uids)].loc[:, ('user', 'item')].sample(10)

    res = lkb.predict(mlb.algo, tf)

    assert len(res) == 10
    assert set(res.user) == set(uids)

    preds = res.set_index(['user', 'item'])
    preds['rating'] = mlb.algo.mean_
    preds['rating'] += mlb.algo.item_offsets_
    preds['rating'] += mlb.algo.user_offsets_
    assert preds.prediction.values == pytest.approx(preds.rating.values)


def test_predict_include_rating(mlb):
    uids = [5, 10]
    tf = None
    # make sure we get both UIDs
    while tf is None or len(set(tf.user)) < 2:
        tf = mlb.ratings[mlb.ratings.user.isin(uids)].loc[:, ('user', 'item', 'rating')].sample(10)

    res = lkb.predict(mlb.algo, tf)

    assert len(res) == 10
    assert set(res.user) == set(uids)

    preds = res.set_index(['user', 'item'])
    preds['expected'] = mlb.algo.mean_
    preds['expected'] += mlb.algo.item_offsets_
    preds['expected'] += mlb.algo.user_offsets_
    assert preds.prediction.values == pytest.approx(preds.expected.values)

    urv = mlb.ratings.set_index(['user', 'item'])
    assert all(preds.rating.values == urv.loc[preds.index, :].rating.values)


@pytest.mark.skipif(not lktu.ml100k.available, reason='ML-100K required')
@pytest.mark.eval
@pytest.mark.parametrize('ncpus', [None, 1, 2])
def test_bias_batch_predict(ncpus):
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.ratings

    algo = basic.Bias(damping=5)

    def eval(train, test):
        _log.info('running training')
        algo.fit(train)
        _log.info('testing %d users', test.user.nunique())
        recs = batch.predict(algo, test, n_jobs=ncpus)
        return recs

    preds = pd.concat((eval(train, test)
                       for (train, test)
                       in xf.partition_users(ratings, 5, xf.SampleFrac(0.2))))

    _log.info('analyzing predictions')
    rmse = pm.rmse(preds.prediction, preds.rating)
    _log.info('RMSE is %f', rmse)
    assert rmse == pytest.approx(0.95, abs=0.1)
