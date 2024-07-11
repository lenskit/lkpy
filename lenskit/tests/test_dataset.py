"""
Tests for the Dataset class.
"""

import numpy as np
import pandas as pd
from pyprojroot import here

from pytest import fixture, raises

from lenskit.data import Dataset, from_interactions_df
from lenskit.data.tables import NumpyUserItemTable, TorchUserItemTable


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
    assert np.all(ml_ds.user_id([1, 5, 23]) == users[[1, 5, 23]])
    assert np.all(ml_ds.user_id(np.array([1, 5, 23])) == users[[1, 5, 23]])


def test_item_id_single(ml_ds: Dataset):
    items = ml_ds.item_vocab
    assert ml_ds.item_id(0) == items[0]
    assert ml_ds.item_id(ml_ds.item_count - 1) == items[-1]
    assert ml_ds.item_id(50) == items[50]


def test_item_id_many(ml_ds: Dataset):
    items = ml_ds.item_vocab
    assert np.all(ml_ds.item_id([1, 5, 23]) == items[[1, 5, 23]])
    assert np.all(ml_ds.item_id(np.array([1, 5, 23])) == items[[1, 5, 23]])


def test_user_num_single(ml_ds: Dataset):
    users = ml_ds.user_vocab
    assert ml_ds.user_num(users[0]) == 0
    assert ml_ds.user_num(users[50]) == 50


def test_user_num_many(ml_ds: Dataset):
    users = ml_ds.user_vocab
    assert np.all(ml_ds.user_num(users[[1, 5, 23]]) == [1, 5, 23])
    assert np.all(ml_ds.user_num(list(users[[1, 5, 23]])) == [1, 5, 23])


def test_user_num_missing_error(ml_ds: Dataset):
    with raises(KeyError):
        ml_ds.user_num(-402, missing="error")


def test_user_num_missing_negative(ml_ds: Dataset):
    assert ml_ds.user_num(-402, missing="negative") == -1


def test_user_num_missing_omit(ml_ds: Dataset):
    user = ml_ds.user_vocab[5]
    series = ml_ds.user_num([user, -402], missing="omit")
    assert len(series) == 1
    assert series.loc[user] == 5


def test_user_num_missing_vector_negative(ml_ds: Dataset):
    u1 = ml_ds.user_vocab[5]
    u2 = ml_ds.user_vocab[100]
    res = ml_ds.user_num([u1, -402, u2], missing="negative")
    assert len(res) == 3
    assert np.all(res == [5, -1, 100])


def test_user_num_missing_vector_error(ml_ds: Dataset):
    u1 = ml_ds.user_vocab[5]
    u2 = ml_ds.user_vocab[100]
    with raises(KeyError):
        ml_ds.user_num([u1, -402, u2], missing="error")


def test_item_num_single(ml_ds: Dataset):
    items = ml_ds.item_vocab
    assert ml_ds.item_num(items[0]) == 0
    assert ml_ds.item_num(items[50]) == 50


def test_item_num_many(ml_ds: Dataset):
    items = ml_ds.item_vocab
    assert np.all(ml_ds.item_num(items[[1, 5, 23]]) == [1, 5, 23])
    assert np.all(ml_ds.item_num(list(items[[1, 5, 23]])) == [1, 5, 23])


def test_item_num_missing_error(ml_ds: Dataset):
    with raises(KeyError):
        ml_ds.item_num(-402, missing="error")


def test_pandas_log_defaults(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    int_df = ml_ds.interaction_log(format="pandas")
    assert isinstance(int_df, pd.DataFrame)
    # we should have exactly the 4 expected columns
    assert len(int_df.columns) == 4
    assert "user_num" in int_df.columns
    assert "item_num" in int_df.columns
    assert "rating" in int_df.columns
    assert "timestamp" in int_df.columns

    # the interact
    int_df = int_df.sort_values(["user_num", "item_num"])
    uids = ml_ds.user_id(int_df["user_num"])
    iids = ml_ds.user_id(int_df["item_num"])

    ml_df = ml_ratings.sort_values(["userId", "movieId"])
    assert np.all(uids == ml_df["userId"])
    assert np.all(iids == ml_df["movieId"])
    assert np.all(int_df["rating"] == ml_df["rating"])
    assert np.all(int_df["timestamp"] == ml_df["timestamp"])

    # and the total length
    assert len(int_df) == len(ml_ratings)


def test_pandas_log_ids(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    int_df = ml_ds.interaction_log(format="pandas", original_ids=True)
    assert isinstance(int_df, pd.DataFrame)
    # we should have exactly the 4 expected columns
    assert len(int_df.columns) == 4
    assert "user_id" in int_df.columns
    assert "item_id" in int_df.columns
    assert "rating" in int_df.columns
    assert "timestamp" in int_df.columns

    # the interact
    int_df = int_df.sort_values(["user_id", "item_id"])

    ml_df = ml_ratings.sort_values(["userId", "movieId"])
    assert np.all(int_df["user_id"] == ml_df["userId"])
    assert np.all(int_df["item_id"] == ml_df["movieId"])
    assert np.all(int_df["rating"] == ml_df["rating"])
    assert np.all(int_df["timestamp"] == ml_df["timestamp"])

    # and the total length
    assert len(int_df) == len(ml_ratings)


def test_numpy_log_defaults(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_log(format="numpy")
    assert isinstance(log, NumpyUserItemTable)
    assert log.ratings is not None
    assert log.timestamps is not None

    # and the total length
    assert len(log.user_nums) == len(ml_ratings)
    assert len(log.item_nums) == len(ml_ratings)
    assert len(log.ratings) == len(ml_ratings)
    assert len(log.timestamps) == len(ml_ratings)


def test_torch_log_defaults(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_log(format="torch")
    assert isinstance(log, TorchUserItemTable)
    assert log.ratings is not None
    assert log.timestamps is not None

    # and the total length
    assert len(log.user_nums) == len(ml_ratings)
    assert len(log.item_nums) == len(ml_ratings)
    assert len(log.ratings) == len(ml_ratings)
    assert len(log.timestamps) == len(ml_ratings)
