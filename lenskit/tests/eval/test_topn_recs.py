# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from pytest import approx

import lenskit.util.test as lktu
from lenskit.algorithms import basic, bias
from lenskit.data import Dataset, from_interactions_df

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


def test_topn_recommend():
    pred = basic.Memorized(simple_df)
    rec = basic.TopN(pred)
    rec.fit(simple_ds)

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
    assert rs.startswith("TopN/")


def test_topn_big(ml_ds: Dataset):
    users = ml_ds.users.ids()
    items = ml_ds.items.ids()
    user_items = ml_ds.interaction_matrix("pandas", original_ids=True).set_index("user_id").item_id

    algo = basic.TopN(bias.Bias())
    a2 = algo.fit(ml_ds)
    assert a2 is algo

    # test 100 random users
    for u in np.random.choice(users, 100, False):
        recs = algo.recommend(u, 100)
        assert len(recs) == 100
        rated = user_items.loc[u]
        assert all(~recs["item"].isin(rated))
        unrated = np.setdiff1d(items, rated)
        scores = algo.predictor.predict_for_user(u, unrated)
        top = scores.nlargest(100)
        assert top.values == approx(recs.score.values)
