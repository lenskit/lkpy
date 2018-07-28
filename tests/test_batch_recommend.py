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


def test_recommend_single(mlb):
    res = lkb.recommend(mlb.algo, mlb.model, [1], None, {1: [31]})

    assert len(res) == 1
    assert all(res['user'] == 1)
    assert all(res['rank'] == 1)
    assert list(res.columns) == ['user', 'rank', 'item', 'score']

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
    assert list(res.columns) == ['user', 'rank', 'item', 'score']
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
