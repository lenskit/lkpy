# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Test the data type utilities in splits.
"""

import numpy as np
import pandas as pd

from lenskit.data.bulk import dict_from_df, iter_item_lists


def test_dict_from_df(rng, ml_ratings: pd.DataFrame):
    ml_ratings = ml_ratings.rename(columns={"user": "user_id", "item": "item_id"})
    users = dict_from_df(ml_ratings)
    assert len(users) == ml_ratings["user_id"].nunique()
    assert set(users.keys()) == set(ml_ratings["user_id"])

    for uid in rng.choice(ml_ratings["user_id"].unique(), 25):
        items = users[uid]
        udf = ml_ratings[ml_ratings["user_id"] == uid]
        assert len(items) == len(udf)
        assert np.all(np.unique(items.ids()) == np.unique(udf["item_id"]))

    tot = sum(len(il) for il in users.values())
    assert tot == len(ml_ratings)


def test_iter_df(ml_ratings: pd.DataFrame):
    counts = {}
    for key, il in iter_item_lists(ml_ratings):
        counts[key] = len(il)

    counts = pd.Series(counts)
    in_counts = ml_ratings.value_counts("user_id")
    assert len(counts) == len(in_counts)
    counts, in_counts = counts.align(in_counts, join="outer")
    assert np.all(counts == in_counts)
