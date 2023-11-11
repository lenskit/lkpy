from lenskit.algorithms import basic
import lenskit.util.test as lktu

import pandas as pd
import numpy as np

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)


def test_empty():
    sel = basic.EmptyCandidateSelector()

    res = sel.candidates(42)
    assert res.tolist() == []


def test_all():
    sel = basic.AllItemsCandidateSelector()
    sel.fit(simple_df)

    assert set(sel.candidates(10)) == set([1, 2, 3])
    assert set(sel.candidates(5)) == set([1, 2, 3])


def test_unrated_selector():
    sel = basic.UnratedItemCandidateSelector()
    s2 = sel.fit(simple_df)
    assert s2 is sel

    assert set(sel.candidates(10)) == set([3])
    assert set(sel.candidates(12)) == set([3, 2])
    assert set(sel.candidates(11)) == set([1, 2, 3])


def test_unrated_override():
    sel = basic.UnratedItemCandidateSelector()
    sel.fit(simple_df)

    assert set(sel.candidates(10, [2])) == set([1, 3])


def test_unrated_big():
    ratings = lktu.ml_test.ratings
    users = ratings.user.unique()
    items = ratings.item.unique()
    user_items = ratings.set_index("user").item

    sel = basic.UnratedItemCandidateSelector()
    s2 = sel.fit(ratings)
    assert s2 is sel

    # test 100 random users
    for u in np.random.choice(users, 100, False):
        candidates = sel.candidates(u)
        candidates = pd.Series(candidates)
        uis = user_items.loc[u]
        assert len(uis) + len(candidates) == len(items)
        assert candidates.nunique() == len(candidates)
        assert all(~candidates.isin(uis))
