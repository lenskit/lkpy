from lenskit.algorithms import basic
from lenskit.algorithms import bias

import pandas as pd
import numpy as np

import lenskit.util.test as lktu
from pytest import approx

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


def test_topn_recommend():
    pred = basic.Memorized(simple_df)
    rec = basic.TopN(pred)
    rec.fit(simple_df)

    rec10 = rec.recommend(10, candidates=[1, 2])
    assert all(rec10.item == [2, 1])
    assert all(rec10.score == [5, 4])

    rec2 = rec.recommend(12, candidates=[1, 2])
    assert len(rec2) == 1
    assert all(rec2.item == [1])
    assert all(rec2.score == [3])

    rec10 = rec.recommend(10, n=1, candidates=[1, 2])
    assert len(rec10) == 1
    assert all(rec10.item == [2])
    assert all(rec10.score == [5])


def test_topn_config():
    pred = basic.Memorized(simple_df)
    rec = basic.TopN(pred)

    rs = str(rec)
    assert rs.startswith('TopN/')


def test_topn_big():
    ratings = lktu.ml_test.ratings
    users = ratings.user.unique()
    items = ratings.item.unique()
    user_items = ratings.set_index('user').item

    algo = basic.TopN(bias.Bias())
    a2 = algo.fit(ratings)
    assert a2 is algo

    # test 100 random users
    for u in np.random.choice(users, 100, False):
        recs = algo.recommend(u, 100)
        assert len(recs) == 100
        rated = user_items.loc[u]
        assert all(~recs['item'].isin(rated))
        unrated = np.setdiff1d(items, rated)
        scores = algo.predictor.predict_for_user(u, unrated)
        top = scores.nlargest(100)
        assert top.values == approx(recs.score.values)
