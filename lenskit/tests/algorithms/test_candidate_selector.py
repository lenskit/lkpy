# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

import lenskit.util.test as lktu
from lenskit.algorithms import basic
from lenskit.data.dataset import Dataset, from_interactions_df

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


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
    s2 = sel.fit(simple_ds)
    assert s2 is sel

    print(sel.items_)
    print(sel.users_)
    print(sel.user_items_)

    assert set(sel.candidates(10)) == set([3])
    assert set(sel.candidates(12)) == set([3, 2])
    assert set(sel.candidates(11)) == set([1, 2, 3])


def test_unrated_override():
    sel = basic.UnratedItemCandidateSelector()
    sel.fit(simple_ds)

    assert set(sel.candidates(10, [2])) == set([1, 3])


def test_unrated_big(ml_ds: Dataset):
    users = ml_ds.users.ids()
    items = ml_ds.items.ids()
    user_items = ml_ds.interaction_matrix("pandas", original_ids=True).set_index("user_id").item_id

    sel = basic.UnratedItemCandidateSelector()
    s2 = sel.fit(ml_ds)
    assert s2 is sel

    # test 100 random users
    for u in np.random.choice(users, 100, False):
        candidates = sel.candidates(u)
        candidates = pd.Series(candidates)
        uis = user_items.loc[u]
        assert len(uis) + len(candidates) == len(items)
        assert candidates.nunique() == len(candidates)
        assert all(~candidates.isin(uis))
