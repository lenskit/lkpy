# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

import lenskit.util.test as lktu
from lenskit.algorithms import basic
from lenskit.data import Dataset, from_interactions_df
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)


def test_memorized():
    algo = basic.Memorized(simple_df)

    preds = algo.predict_for_user(10, [1, 2])
    assert set(preds.index) == set([1, 2])
    assert all(preds == pd.Series({1: 4.0, 2: 5.0}))

    preds = algo.predict_for_user(12, [1, 3])
    assert set(preds.index) == set([1, 3])
    assert preds.loc[1] == 3.0
    assert np.isnan(preds.loc[3])


def test_memorized_batch():
    algo = basic.Memorized(simple_df)

    preds = algo.predict(pd.DataFrame({"user": [10, 10, 12], "item": [1, 2, 1]}))
    assert isinstance(preds, pd.Series)
    assert preds.name == "prediction"
    assert set(preds.index) == set([0, 1, 2])
    assert all(preds == [4.0, 5.0, 3.0])


def test_memorized_batch_ord():
    algo = basic.Memorized(simple_df)

    preds = algo.predict(pd.DataFrame({"user": [10, 12, 10], "item": [1, 1, 2]}))
    assert set(preds.index) == set([0, 1, 2])
    assert all(preds == [4.0, 3.0, 5.0])


def test_memorized_batch_missing():
    algo = basic.Memorized(simple_df)

    preds = algo.predict(pd.DataFrame({"user": [10, 12, 12], "item": [1, 1, 3]}))
    assert set(preds.index) == set([0, 1, 2])
    assert all(preds.iloc[:2] == [4.0, 3.0])
    assert np.isnan(preds.iloc[2])


def test_memorized_batch_keep_index():
    algo = basic.Memorized(simple_df)

    query = pd.DataFrame(
        {"user": [10, 10, 12], "item": [1, 2, 1]}, index=np.random.choice(np.arange(10), 3, False)
    )
    preds = algo.predict(query)
    assert all(preds.index == query.index)
    assert all(preds == [4.0, 5.0, 3.0])


def test_random(ml_ds: Dataset):
    # test case: no seed
    algo = basic.Random()
    model = algo.fit(ml_ds)

    assert model is not None

    recs1 = algo.recommend(2038, 100)
    recs2 = algo.recommend(2028, 100)
    assert len(recs1) == 100
    assert len(recs2) == 100
    # with very high probabilities
    assert set(recs1["item"]) != set(recs2["item"])

    recs_all = algo.recommend(2038)
    assert len(recs_all) == ml_ds.item_count
    assert set(ml_ds.items.ids()) == set(recs_all["item"])


def test_random_derive_seed(ml_ds: Dataset):
    algo = basic.Random(rng_spec="user")
    model = algo.fit(ml_ds)

    assert model is not None

    recs1 = algo.recommend(2038, 100)
    recs2 = algo.recommend(2028, 100)
    assert len(recs1) == 100
    assert len(recs2) == 100
    # with very high probabilities
    assert set(recs1["item"]) != set(recs2["item"])

    recs_all = algo.recommend(2038)
    assert len(recs_all) == ml_ds.item_count
    assert set(ml_ds.items.ids()) == set(recs_all["item"])


def test_random_rec_from_candidates(ml_ds: Dataset):
    algo = basic.Random()
    items = ml_ds.items.ids()
    users = ml_ds.users.ids()
    user1, user2 = np.random.choice(users, size=2, replace=False)
    algo.fit(ml_ds)

    # recommend from candidates
    candidates = np.random.choice(items, 500, replace=False)
    recs1 = algo.recommend(user1, 100, candidates=candidates)
    recs2 = algo.recommend(user2, 100, candidates=candidates)
    assert len(recs1) == 100
    assert len(recs2) == 100


def test_knownrating():
    algo = basic.KnownRating()
    algo.fit(simple_ds)

    preds = algo.predict_for_user(10, [1, 2])
    assert set(preds.index) == set([1, 2])
    assert all(preds == pd.Series({1: 4.0, 2: 5.0}))

    preds = algo.predict_for_user(12, [1, 3])
    assert set(preds.index) == set([1, 3])
    assert preds.loc[1] == 3.0
    assert np.isnan(preds.loc[3])


def test_knownrating_batch_missing():
    algo = basic.KnownRating()
    algo.fit(simple_ds)

    preds = algo.predict(pd.DataFrame({"user": [10, 12, 12], "item": [1, 1, 3]}))
    assert set(preds.index) == set([0, 1, 2])
    assert all(preds.iloc[:2] == [4.0, 3.0])
    assert np.isnan(preds.iloc[2])
