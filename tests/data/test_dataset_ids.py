# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for the Dataset class.
"""

import numpy as np
import pandas as pd

from pytest import raises

from lenskit.data import Dataset, from_interactions_df
from lenskit.testing import ml_ds, ml_ratings  # noqa: F401


def test_from_ratings_default_names(ml_ratings: pd.DataFrame):
    ratings = ml_ratings
    ds = from_interactions_df(ratings)
    assert ds.item_count == ratings["item_id"].nunique()
    assert ds.user_count == ratings["user_id"].nunique()


def test_from_ratings_nosuffix(ml_ratings: pd.DataFrame):
    ratings = ml_ratings.rename(columns={"user_id": "user", "item_id": "item"})
    ds = from_interactions_df(ratings)
    assert ds.item_count == ratings["item"].nunique()
    assert ds.user_count == ratings["user"].nunique()


def test_from_ratings_names_upper(ml_ratings: pd.DataFrame):
    ratings = ml_ratings.rename(columns={"user_id": "USER", "item_id": "ITEM"})
    ds = from_interactions_df(ratings)
    assert ds.item_count == ratings["ITEM"].nunique()
    assert ds.user_count == ratings["USER"].nunique()


def test_user_id_single(ml_ds: Dataset):
    users = ml_ds.users.index
    assert ml_ds.users.id(0) == users[0]
    assert ml_ds.users.id(ml_ds.user_count - 1) == users[-1]
    assert ml_ds.users.id(50) == users[50]


def test_user_id_many(ml_ds: Dataset):
    users = ml_ds.users.index
    assert np.all(ml_ds.users.ids([1, 5, 23]) == users[[1, 5, 23]])
    assert np.all(ml_ds.users.ids(np.array([1, 5, 23])) == users[[1, 5, 23]])


def test_item_id_single(ml_ds: Dataset):
    items = ml_ds.items.index
    assert ml_ds.items.id(0) == items[0]
    assert ml_ds.items.id(ml_ds.item_count - 1) == items[-1]
    assert ml_ds.items.id(50) == items[50]


def test_item_id_many(ml_ds: Dataset):
    items = ml_ds.items.index
    assert np.all(ml_ds.items.ids([1, 5, 23]) == items[[1, 5, 23]])
    assert np.all(ml_ds.items.ids(np.array([1, 5, 23])) == items[[1, 5, 23]])


def test_user_num_single(ml_ds: Dataset):
    users = ml_ds.users.index
    assert ml_ds.users.number(users[0]) == 0
    assert ml_ds.users.number(users[50]) == 50


def test_user_num_many(ml_ds: Dataset):
    users = ml_ds.users.index
    assert np.all(ml_ds.users.numbers(users[[1, 5, 23]]) == [1, 5, 23])
    assert np.all(ml_ds.users.numbers(list(users[[1, 5, 23]])) == [1, 5, 23])


def test_user_num_missing_error(ml_ds: Dataset):
    with raises(KeyError):
        ml_ds.users.number(-402, missing="error")


def test_user_num_missing_negative(ml_ds: Dataset):
    assert ml_ds.users.number(-402, missing=None) is None


def test_user_num_missing_vector_negative(ml_ds: Dataset):
    u1 = ml_ds.users.index[5]
    u2 = ml_ds.users.index[100]
    res = ml_ds.users.numbers([u1, -402, u2], missing="negative")
    assert len(res) == 3
    assert np.all(res == [5, -1, 100])


def test_user_num_missing_vector_error(ml_ds: Dataset):
    u1 = ml_ds.users.index[5]
    u2 = ml_ds.users.index[100]
    with raises(KeyError):
        ml_ds.users.numbers([u1, -402, u2], missing="error")


def test_item_num_single(ml_ds: Dataset):
    items = ml_ds.items.index
    assert ml_ds.items.number(items[0]) == 0
    assert ml_ds.items.number(items[50]) == 50


def test_item_num_many(ml_ds: Dataset):
    items = ml_ds.items.index
    assert np.all(ml_ds.items.numbers(items[[1, 5, 23]]) == [1, 5, 23])
    assert np.all(ml_ds.items.numbers(list(items[[1, 5, 23]])) == [1, 5, 23])


def test_item_num_missing_error(ml_ds: Dataset):
    with raises(KeyError):
        ml_ds.items.number(-402, missing="error")
