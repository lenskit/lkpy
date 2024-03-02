# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.algorithms import basic

import pandas as pd
import numpy as np
import pickle

import lenskit.util.test as lktu

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)


def test_popular():
    algo = basic.Popular()
    algo.fit(lktu.ml_test.ratings)
    counts = lktu.ml_test.ratings.groupby("item").user.count()
    counts = counts.nlargest(100)

    assert algo.item_pop_.max() == counts.max()

    recs = algo.recommend(2038, 100)
    assert len(recs) == 100
    assert all(np.diff(recs.score) <= 0)

    assert recs.score.iloc[0] == counts.max()
    # the 10 most popular should be the same
    assert all(counts.index[:10] == recs.item[:10])


def test_popular_excludes_rated():
    algo = basic.Popular()
    algo.fit(lktu.ml_test.ratings)
    counts = lktu.ml_test.ratings.groupby("item").user.count()
    counts = counts.nlargest(100)

    recs = algo.recommend(100, 100)
    assert len(recs) == 100
    assert all(np.diff(recs.score) <= 0)

    # make sure we didn't recommend anything the user likes
    ratings = lktu.ml_test.ratings
    urates = ratings.set_index(["user", "item"])
    urates = urates.loc[100, :]
    match = recs.join(urates, on="item", how="inner")
    assert len(match) == 0


def test_pop_candidates():
    algo = basic.Popular()
    algo.fit(lktu.ml_test.ratings)
    counts = lktu.ml_test.ratings.groupby("item").user.count()
    items = lktu.ml_test.ratings.item.unique()

    assert algo.item_pop_.max() == counts.max()

    candidates = np.random.choice(items, 500, replace=False)

    recs = algo.recommend(2038, 100, candidates)
    assert len(recs) == 100
    assert all(np.diff(recs.score) <= 0)

    ccs = counts.loc[candidates]
    ccs = ccs.sort_values(ascending=False)

    assert recs.score.iloc[0] == ccs.max()
    equiv = ccs[ccs == ccs.max()]
    assert recs.item.iloc[0] in equiv.index


def test_pop_save_load():
    original = basic.Popular()
    original.fit(lktu.ml_test.ratings)

    mod = pickle.dumps(original)
    algo = pickle.loads(mod)

    counts = lktu.ml_test.ratings.groupby("item").user.count()
    counts = counts.nlargest(100)

    assert algo.item_pop_.max() == counts.max()

    recs = algo.recommend(2038, 100)
    assert len(recs) == 100
    assert all(np.diff(recs.score) <= 0)

    assert recs.score.iloc[0] == counts.max()
    # the 10 most popular should be the same
    assert all(counts.index[:10] == recs.item[:10])


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
