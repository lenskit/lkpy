# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle

import pandas as pd

import lenskit.util.test as lktu
from lenskit.algorithms import basic

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)


def test_popscore_quantile(rng):
    algo = basic.PopScore()
    algo.fit(lktu.ml_test.ratings)

    assert algo.item_scores_.max() == 1.0

    counts = lktu.ml_test.ratings.groupby("item").user.count()
    counts = counts.sort_values()

    winner = counts.index[-1]
    assert algo.item_scores_.loc[winner] == 1.0


def test_popscore_rank(rng):
    algo = basic.PopScore("rank")
    algo.fit(lktu.ml_test.ratings)

    counts = lktu.ml_test.ratings.groupby("item").user.count()
    counts = counts.sort_values()

    assert algo.item_scores_.max() == len(counts)

    winner = counts.index[-1]
    assert algo.item_scores_.loc[winner] == len(counts)


def test_popscore_counts(rng):
    algo = basic.PopScore("count")
    algo.fit(lktu.ml_test.ratings)

    counts = lktu.ml_test.ratings.groupby("item").user.count()

    scores, counts = algo.item_scores_.align(counts)
    assert all(scores == counts)

    items = rng.choice(counts.index, 100)
    scores = algo.predict_for_user(20, items)
    assert all(scores == counts.loc[items])


def test_popscore_save_load():
    original = basic.PopScore()
    original.fit(lktu.ml_test.ratings)

    mod = pickle.dumps(original)
    algo = pickle.loads(mod)

    assert all(algo.item_scores_ == original.item_scores_)
