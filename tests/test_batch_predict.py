import pytest

from collections import namedtuple
from functools import partial
import pandas as pd
import numpy as np

import lk_test_utils as lktu

from lenskit.algorithms.basic import Bias
import lenskit.batch as lkb

MLB = namedtuple('MLB', ['ratings', 'algo', 'model'])
MLB.predictor = property(lambda mlb: partial(mlb.algo.predict, mlb.model))


@pytest.fixture
def mlb():
    ratings = lktu.ml_pandas.renamed.ratings
    algo = Bias()
    model = algo.train(ratings)
    return MLB(ratings, algo, model)


def test_predict_single(mlb):
    tf = pd.DataFrame({'user': [1], 'item': [31]})
    res = lkb.predict(mlb.predictor, tf)

    assert len(res) == 1
    assert all(res.user == 1)
    assert list(res.columns) == ['user', 'item', 'prediction']
    assert all(res.item == 31)

    expected = mlb.model.mean + mlb.model.items.loc[31] + mlb.model.users.loc[1]
    assert res.prediction.iloc[0] == pytest.approx(expected)


def test_predict_single_model(mlb):
    tf = pd.DataFrame({'user': [1], 'item': [31]})
    res = lkb.predict(mlb.algo, tf, mlb.model)

    assert len(res) == 1
    assert all(res.user == 1)
    assert list(res.columns) == ['user', 'item', 'prediction']
    assert all(res.item == 31)

    expected = mlb.model.mean + mlb.model.items.loc[31] + mlb.model.users.loc[1]
    assert res.prediction.iloc[0] == pytest.approx(expected)


def test_predict_user(mlb):
    uid = 5
    urates = mlb.ratings[mlb.ratings.user == uid]

    test_rated = urates.item.sample(5)
    unrated = np.setdiff1d(mlb.ratings.item.unique(), urates.item.values)
    test_unrated = np.random.choice(unrated, 10, replace=False)
    test_items = pd.concat([test_rated, pd.Series(test_unrated)])

    tf = pd.DataFrame({'user': uid, 'item': test_items})
    res = lkb.predict(mlb.predictor, tf)

    assert len(res) == 15
    assert list(res.columns) == ['user', 'item', 'prediction']
    assert all(res.user == uid)
    assert set(res.item) == set(test_items)

    # did we get the right predictions?
    preds = res.set_index(['user', 'item'])
    preds['rating'] = mlb.model.mean
    preds['rating'] += mlb.model.items
    preds['rating'] += mlb.model.users.loc[uid]
    assert preds.prediction.values == pytest.approx(preds.rating.values)


def test_predict_two_users(mlb):
    uids = [5, 10]
    tf = None
    # make sure we get both UIDs
    while tf is None or len(set(tf.user)) < 2:
        tf = mlb.ratings[mlb.ratings.user.isin(uids)].loc[:, ('user', 'item')].sample(10)

    res = lkb.predict(mlb.predictor, tf)

    assert len(res) == 10
    assert set(res.user) == set(uids)

    preds = res.set_index(['user', 'item'])
    preds['rating'] = mlb.model.mean
    preds['rating'] += mlb.model.items
    preds['rating'] += mlb.model.users
    assert preds.prediction.values == pytest.approx(preds.rating.values)


def test_predict_include_rating(mlb):
    uids = [5, 10]
    tf = None
    # make sure we get both UIDs
    while tf is None or len(set(tf.user)) < 2:
        tf = mlb.ratings[mlb.ratings.user.isin(uids)].loc[:, ('user', 'item', 'rating')].sample(10)

    res = lkb.predict(mlb.predictor, tf)

    assert len(res) == 10
    assert set(res.user) == set(uids)

    preds = res.set_index(['user', 'item'])
    preds['expected'] = mlb.model.mean
    preds['expected'] += mlb.model.items
    preds['expected'] += mlb.model.users
    assert preds.prediction.values == pytest.approx(preds.expected.values)

    urv = mlb.ratings.set_index(['user', 'item'])
    assert all(preds.rating.values == urv.loc[preds.index, :].rating.values)
