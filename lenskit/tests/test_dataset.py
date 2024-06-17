"""
Tests for the Dataset class.
"""

import numpy as np
import pandas as pd
from pyprojroot import here

from pytest import fixture

from lenskit.data import Dataset, from_interactions_df


@fixture(scope="module")
def ml_ratings():
    """
    Fixture to load the test MovieLens ratings.
    """
    path = here("data/ml-latest-small")
    yield pd.read_csv(path / "ratings.csv")


@fixture
def ml_ds(ml_ratings: pd.DataFrame):
    return from_interactions_df(ml_ratings, item_col="movieId")


def test_from_ratings_default_names(ml_ratings: pd.DataFrame):
    ratings = ml_ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    ds = from_interactions_df(ratings)
    assert ds.item_count == ratings["item_id"].nunique()
    assert ds.user_count == ratings["user_id"].nunique()


def test_from_ratings_nosuffix(ml_ratings: pd.DataFrame):
    ratings = ml_ratings.rename(columns={"userId": "user", "movieId": "item"})
    ds = from_interactions_df(ratings)
    assert ds.item_count == ratings["item"].nunique()
    assert ds.user_count == ratings["user"].nunique()


def test_from_ratings_names_upper(ml_ratings: pd.DataFrame):
    ratings = ml_ratings.rename(columns={"userId": "USER", "movieId": "ITEM"})
    ds = from_interactions_df(ratings)
    assert ds.item_count == ratings["ITEM"].nunique()
    assert ds.user_count == ratings["USER"].nunique()


def test_user_id_single(ml_ds: Dataset):
    users = ml_ds.user_vocab
    assert ml_ds.user_id(0) == users[0]
    assert ml_ds.user_id(ml_ds.user_count - 1) == users[-1]
    assert ml_ds.user_id(50) == users[50]


def test_user_id_many(ml_ds: Dataset):
    users = ml_ds.user_vocab
    assert ml_ds.user_id([1, 5, 23]) == users[[1, 5, 23]]
    assert ml_ds.user_id(np.array([1, 5, 23])) == users[[1, 5, 23]]


def test_item_id_single(ml_ds: Dataset):
    items = ml_ds.item_vocab
    assert ml_ds.item_id(0) == items[0]
    assert ml_ds.item_id(ml_ds.item_count - 1) == items[-1]
    assert ml_ds.item_id(50) == items[50]


def test_item_id_many(ml_ds: Dataset):
    items = ml_ds.item_vocab
    assert ml_ds.item_id([1, 5, 23]) == items[[1, 5, 23]]
    assert ml_ds.item_id(np.array([1, 5, 23])) == items[[1, 5, 23]]
