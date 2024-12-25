# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for the Dataset class.
"""

import numpy as np
import pandas as pd

from lenskit.data import Dataset
from lenskit.data.tables import NumpyUserItemTable, TorchUserItemTable
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401


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
    uids = ml_ds.users.ids(int_df["user_num"])
    iids = ml_ds.items.ids(int_df["item_num"])

    ml_df = ml_ratings.sort_values(["user_id", "item_id"])
    assert np.all(uids == ml_df["user_id"])
    assert np.all(iids == ml_df["item_id"])
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

    ml_df = ml_ratings.sort_values(["user_id", "item_id"])
    assert np.all(int_df["user_id"] == ml_df["user_id"])
    assert np.all(int_df["item_id"] == ml_df["item_id"])
    assert np.all(int_df["rating"] == ml_df["rating"])
    assert np.all(int_df["timestamp"] == ml_df["timestamp"])

    # and the total length
    assert len(int_df) == len(ml_ratings)


def test_pandas_log_no_ts(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    int_df = ml_ds.interaction_log(format="pandas", fields=["rating"])
    assert isinstance(int_df, pd.DataFrame)
    # we should have exactly the 4 expected columns
    assert len(int_df.columns) == 3
    assert "user_num" in int_df.columns
    assert "item_num" in int_df.columns
    assert "rating" in int_df.columns

    # the interact
    int_df = int_df.sort_values(["user_num", "item_num"])
    uids = ml_ds.users.ids(int_df["user_num"])
    iids = ml_ds.items.ids(int_df["item_num"])

    ml_df = ml_ratings.sort_values(["user_id", "item_id"])
    assert np.all(uids == ml_df["user_id"])
    assert np.all(iids == ml_df["item_id"])
    assert np.all(int_df["rating"] == ml_df["rating"])

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
