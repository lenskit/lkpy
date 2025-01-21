# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for the Dataset class.
"""

from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
from numpy.typing import ArrayLike

from pytest import mark, raises

from lenskit.data import Dataset, FieldError, from_interactions_df
from lenskit.data.matrix import CSRStructure, MatrixDataset
from lenskit.testing import ml_ds, ml_ratings  # noqa: F401


def _check_user_offset_counts(ml_ds: Dataset, ml_ratings: pd.DataFrame, offsets: ArrayLike):
    user_counts = ml_ratings["user_id"].value_counts().reindex(ml_ds.users.index)
    row_lens = np.diff(offsets)
    assert np.all(row_lens == user_counts)


def _check_user_number_counts(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    users, counts = np.unique(nums, return_counts=True)
    user_counts = ml_ratings["user_id"].value_counts().reindex(ml_ds.users.ids(users))
    assert np.all(counts == user_counts)


def _check_item_number_counts(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    items, counts = np.unique(nums, return_counts=True)
    item_counts = ml_ratings["item_id"].value_counts().reindex(ml_ds.items.ids(items))
    assert np.all(counts == item_counts)


def _check_user_ids(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["user_id", "item_id"])
    assert np.all(ml_ds.users.ids(np.asarray(nums)) == ml_ratings["user_id"])


def _check_item_ids(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["user_id", "item_id"])
    assert np.all(ml_ds.items.ids(np.asarray(nums)) == ml_ratings["item_id"])


def _check_ratings(ml_ds: Dataset, ml_ratings: pd.DataFrame, rates: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["user_id", "item_id"])
    assert np.all(rates == ml_ratings["rating"])


def _check_timestamp(ml_ds: Dataset, ml_ratings: pd.DataFrame, ts: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["user_id", "item_id"])
    assert np.all(ts == ml_ratings["timestamp"])


def test_matrix_structure(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="structure")
    assert isinstance(log, CSRStructure)
    assert log.nnz == len(ml_ratings)

    assert log.nrows == ml_ratings["user_id"].nunique()
    assert log.ncols == ml_ratings["item_id"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.rowptrs)
    _check_item_number_counts(ml_ds, ml_ratings, log.colinds)
    _check_item_ids(ml_ds, ml_ratings, log.colinds)


def test_matrix_structure_field_fail(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    with raises(ValueError, match="structure does not support fields"):
        ml_ds.interaction_matrix(format="structure", field="timestamp")  # type: ignore


def test_matrix_structure_coo_fail(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    with raises(ValueError, match="unsupported layout"):
        ml_ds.interaction_matrix(format="structure", layout="coo")  # type: ignore


def test_matrix_pandas(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="pandas", field="rating")
    assert isinstance(log, pd.DataFrame)
    assert len(log) == len(ml_ratings)
    assert "rating" in log.columns

    _check_user_number_counts(ml_ds, ml_ratings, log["user_num"])
    _check_user_ids(ml_ds, ml_ratings, log["user_num"])
    _check_item_number_counts(ml_ds, ml_ratings, log["item_num"])
    _check_item_ids(ml_ds, ml_ratings, log["item_num"])
    _check_ratings(ml_ds, ml_ratings, log["rating"])


def test_matrix_pandas_orig_id(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    "Test that Pandas can return original IDs."
    log = ml_ds.interaction_matrix(format="pandas", field="rating", original_ids=True)
    assert isinstance(log, pd.DataFrame)
    assert len(log) == len(ml_ratings)
    assert "user_id" in log.columns
    assert "item_id" in log.columns

    m2 = ml_ds.interaction_matrix(format="scipy", layout="coo")

    assert np.all(log["user_id"] == ml_ds.users.ids(m2.row))
    assert np.all(log["item_id"] == ml_ds.items.ids(m2.col))

    _check_ratings(ml_ds, ml_ratings, log["rating"])


def test_matrix_pandas_timestamp(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="pandas", field="timestamp")
    assert isinstance(log, pd.DataFrame)
    assert len(log) == len(ml_ratings)

    _check_user_number_counts(ml_ds, ml_ratings, log["user_num"])
    _check_user_ids(ml_ds, ml_ratings, log["user_num"])
    _check_item_number_counts(ml_ds, ml_ratings, log["item_num"])
    _check_item_ids(ml_ds, ml_ratings, log["item_num"])
    _check_timestamp(ml_ds, ml_ratings, log["timestamp"])


def test_matrix_pandas_unknown_field(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    with raises(FieldError, match=r"rating\[playcount\]"):
        ml_ds.interaction_matrix(format="pandas", field="playcount")  # type: ignore


def test_matrix_pandas_indicator(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="pandas")
    assert isinstance(log, pd.DataFrame)
    assert len(log) == len(ml_ratings)

    _check_user_number_counts(ml_ds, ml_ratings, log["user_num"])
    _check_user_ids(ml_ds, ml_ratings, log["user_num"])
    _check_item_number_counts(ml_ds, ml_ratings, log["item_num"])
    _check_item_ids(ml_ds, ml_ratings, log["item_num"])


@mark.parametrize("generation", ["modern", "legacy"])
def test_matrix_scipy_coo(ml_ratings: pd.DataFrame, ml_ds: Dataset, generation):
    log = ml_ds.interaction_matrix(
        format="scipy", layout="coo", field="rating", legacy=generation == "legacy"
    )
    assert isinstance(log, sps.coo_array if generation == "modern" else sps.coo_matrix)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user_id"].nunique()
    assert ncols == ml_ratings["item_id"].nunique()

    assert log.row.dtype == np.int32
    assert log.col.dtype == np.int32
    _check_user_number_counts(ml_ds, ml_ratings, log.row)
    _check_user_ids(ml_ds, ml_ratings, log.row)
    # ensure users are sorted
    assert np.all(np.diff(log.row) >= 0)

    _check_item_number_counts(ml_ds, ml_ratings, log.col)
    _check_item_ids(ml_ds, ml_ratings, log.col)
    _check_ratings(ml_ds, ml_ratings, log.data)


@mark.parametrize("generation", ["modern", "legacy"])
def test_matrix_scipy_csr(ml_ratings: pd.DataFrame, ml_ds: Dataset, generation):
    log = ml_ds.interaction_matrix(format="scipy", field="rating", legacy=generation == "legacy")
    assert isinstance(log, sps.csr_array if generation == "modern" else sps.csr_matrix)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user_id"].nunique()
    assert ncols == ml_ratings["item_id"].nunique()

    assert log.indptr.dtype == np.int32
    assert log.indices.dtype == np.int32
    _check_user_offset_counts(ml_ds, ml_ratings, log.indptr)
    _check_item_number_counts(ml_ds, ml_ratings, log.indices)
    _check_item_ids(ml_ds, ml_ratings, log.indices)
    _check_ratings(ml_ds, ml_ratings, log.data)


@mark.parametrize("generation", ["modern", "legacy"])
def test_matrix_scipy_timestamp(ml_ratings: pd.DataFrame, ml_ds: Dataset, generation):
    log = ml_ds.interaction_matrix(format="scipy", field="timestamp", legacy=generation == "legacy")
    assert isinstance(log, sps.csr_array if generation == "modern" else sps.csr_matrix)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user_id"].nunique()
    assert ncols == ml_ratings["item_id"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.indptr)
    _check_item_number_counts(ml_ds, ml_ratings, log.indices)
    _check_item_ids(ml_ds, ml_ratings, log.indices)
    _check_timestamp(ml_ds, ml_ratings, log.data)


@mark.parametrize("generation", ["modern", "legacy"])
def test_matrix_scipy_indicator(ml_ratings: pd.DataFrame, ml_ds: Dataset, generation):
    log = ml_ds.interaction_matrix(format="scipy", legacy=generation == "legacy")
    assert isinstance(log, sps.csr_array if generation == "modern" else sps.csr_matrix)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user_id"].nunique()
    assert ncols == ml_ratings["item_id"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.indptr)
    _check_item_number_counts(ml_ds, ml_ratings, log.indices)
    _check_item_ids(ml_ds, ml_ratings, log.indices)

    # right rating values
    assert np.all(log.data == 1.0)


def test_matrix_torch_csr(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="torch", field="rating")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse_csr
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = log.shape
    assert nrows == ml_ratings["user_id"].nunique()
    assert ncols == ml_ratings["item_id"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.crow_indices())
    _check_item_number_counts(ml_ds, ml_ratings, log.col_indices())
    _check_item_ids(ml_ds, ml_ratings, log.col_indices())
    _check_ratings(ml_ds, ml_ratings, log.values().numpy())

    assert log.crow_indices().dtype == torch.int32
    assert log.col_indices().dtype == torch.int32


def test_matrix_torch_indicator(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="torch")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse_csr
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = log.shape
    assert nrows == ml_ratings["user_id"].nunique()
    assert ncols == ml_ratings["item_id"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.crow_indices())
    _check_item_number_counts(ml_ds, ml_ratings, log.col_indices())
    _check_item_ids(ml_ds, ml_ratings, log.col_indices())
    assert np.all(log.values().numpy() == 1.0)


def test_matrix_torch_coo(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="torch", layout="coo", field="rating")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user_id"].nunique()
    assert ncols == ml_ratings["item_id"].nunique()

    _check_user_number_counts(ml_ds, ml_ratings, log.indices()[0, :])
    _check_user_ids(ml_ds, ml_ratings, log.indices()[0, :])
    _check_item_number_counts(ml_ds, ml_ratings, log.indices()[1, :])
    _check_item_ids(ml_ds, ml_ratings, log.indices()[1, :])
    _check_ratings(ml_ds, ml_ratings, log.values().numpy())


def test_matrix_torch_timestamp(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="torch", field="timestamp")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse_csr
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = log.shape
    assert nrows == ml_ratings["user_id"].nunique()
    assert ncols == ml_ratings["item_id"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.crow_indices())
    _check_item_number_counts(ml_ds, ml_ratings, log.col_indices())
    _check_item_ids(ml_ds, ml_ratings, log.col_indices())
    _check_timestamp(ml_ds, ml_ratings, log.values().numpy())


def test_matrix_rows_by_id(rng: np.random.Generator, ml_ratings: pd.DataFrame, ml_ds: Dataset):
    users = rng.choice(ml_ds.users.ids(), 50)

    for user in users:
        row = ml_ds.user_row(user)
        assert row is not None
        urows = ml_ratings[ml_ratings["user_id"] == user].sort_values("item_id")
        urows = urows.reset_index(drop=True)
        assert set(row.ids()) == set(urows["item_id"])
        assert np.all(row.numbers() == ml_ds.items.numbers(urows["item_id"]))

        ratings = row.field("rating")
        assert ratings is not None
        assert np.all(ratings == urows["rating"])

        timestamps = row.field("timestamp")
        assert timestamps is not None
        assert np.all(timestamps == urows["timestamp"])

        # we'll quick check additional fields on the item list here
        df = row.to_df()
        assert np.all(df["timestamp"] == urows["timestamp"])


def test_matrix_rows_by_num(rng: np.random.Generator, ml_ratings: pd.DataFrame, ml_ds: Dataset):
    users = rng.choice(ml_ds.user_count, 50)

    rated = set(zip(ml_ratings["user_id"], ml_ratings["item_id"]))
    rdf = ml_ds.interaction_matrix(format="pandas")
    rnums = set(zip(rdf["user_num"], rdf["item_num"]))

    dfi = ml_ratings.set_index(["user_id", "item_id"])

    for user in users:
        uid = ml_ds.users.id(user)
        row = ml_ds.user_row(user_num=user)
        assert row is not None
        assert row is not None
        urows = ml_ratings[ml_ratings["user_id"] == ml_ds.users.id(user)].sort_values("item_id")
        assert set(row.ids()) == set(urows["item_id"])

        assert np.all(row.numbers() == ml_ds.items.numbers(urows["item_id"]))
        assert all((user, ino) in rnums for ino in row.numbers())

        assert np.all(row.ids() == ml_ds.items.ids(row.numbers()))
        assert all((uid, item) in rated for item in row.ids())
        assert all((uid, item) in dfi.index for item in row.ids())

        ratings = row.field("rating")
        assert ratings is not None
        assert np.all(ratings == urows["rating"])

        timestamps = row.field("timestamp")
        assert timestamps is not None
        assert np.all(timestamps == urows["timestamp"])
