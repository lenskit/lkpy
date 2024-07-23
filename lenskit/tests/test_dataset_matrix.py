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

from lenskit.data import Dataset
from lenskit.data.dataset import FieldError, from_interactions_df
from lenskit.data.matrix import CSRStructure
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401


def _check_user_offset_counts(ml_ds: Dataset, ml_ratings: pd.DataFrame, offsets: ArrayLike):
    user_counts = ml_ratings["userId"].value_counts().reindex(ml_ds.users.index)
    row_lens = np.diff(offsets)
    assert np.all(row_lens == user_counts)


def _check_user_number_counts(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    users, counts = np.unique(nums, return_counts=True)
    user_counts = ml_ratings["userId"].value_counts().reindex(ml_ds.users.ids(users))
    assert np.all(counts == user_counts)


def _check_item_number_counts(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    items, counts = np.unique(nums, return_counts=True)
    item_counts = ml_ratings["movieId"].value_counts().reindex(ml_ds.items.ids(items))
    assert np.all(counts == item_counts)


def _check_user_ids(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    assert np.all(ml_ds.users.ids(np.asarray(nums)) == ml_ratings["userId"])


def _check_item_ids(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    assert np.all(ml_ds.items.ids(np.asarray(nums)) == ml_ratings["movieId"])


def _check_ratings(ml_ds: Dataset, ml_ratings: pd.DataFrame, rates: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    assert np.all(rates == ml_ratings["rating"])


def _check_timestamp(ml_ds: Dataset, ml_ratings: pd.DataFrame, ts: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    assert np.all(ts == ml_ratings["timestamp"])


def test_internals(ml_ds: Dataset):
    "Test internal matrix structures"
    assert ml_ds._matrix.user_nums.dtype == np.int32
    assert ml_ds._matrix.user_ptrs.dtype == np.int32
    assert ml_ds._matrix.item_nums.dtype == np.int32


def test_matrix_structure(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="structure")
    assert isinstance(log, CSRStructure)
    assert log.nnz == len(ml_ratings)

    assert log.nrows == ml_ratings["userId"].nunique()
    assert log.ncols == ml_ratings["movieId"].nunique()

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
    assert 'user_id' in log.columns
    assert 'item_id' in log.columns

    m2 = ml_ds.interaction_matrix(format="scipy", layout='coo')

    assert np.all(log['user_id'] == ml_ds.users.ids(m2.row))
    assert np.all(log['item_id'] == ml_ds.items.ids(m2.col))

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


def test_matrix_pandas_csr_fail(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    with raises(ValueError, match="unsupported layout"):
        ml_ds.interaction_matrix(format="pandas", field="rating", layout="csr")  # type: ignore


def test_matrix_pandas_unknown_field(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    with raises(FieldError, match=r"interaction\[playcount\]"):
        ml_ds.interaction_matrix(format="pandas", field="playcount")  # type: ignore


def test_matrix_pandas_indicator(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="pandas")
    assert isinstance(log, pd.DataFrame)
    assert len(log) == len(ml_ratings)

    _check_user_number_counts(ml_ds, ml_ratings, log["user_num"])
    _check_user_ids(ml_ds, ml_ratings, log["user_num"])
    _check_item_number_counts(ml_ds, ml_ratings, log["item_num"])
    _check_item_ids(ml_ds, ml_ratings, log["item_num"])


def test_matrix_pandas_missing_rating(ml_ratings: pd.DataFrame):
    ml_ds = from_interactions_df(ml_ratings[["userId", "movieId", "timestamp"]], item_col="movieId")
    log = ml_ds.interaction_matrix(format="pandas", field="rating")
    assert isinstance(log, pd.DataFrame)
    assert len(log) == len(ml_ratings)

    _check_user_number_counts(ml_ds, ml_ratings, log["user_num"])
    _check_user_ids(ml_ds, ml_ratings, log["user_num"])
    _check_item_number_counts(ml_ds, ml_ratings, log["item_num"])
    _check_item_ids(ml_ds, ml_ratings, log["item_num"])
    assert np.all(log["rating"] == 1.0)


@mark.parametrize("generation", ["modern", "legacy"])
def test_matrix_scipy_coo(ml_ratings: pd.DataFrame, ml_ds: Dataset, generation):
    log = ml_ds.interaction_matrix(
        format="scipy", layout="coo", field="rating", legacy=generation == "legacy"
    )
    assert isinstance(log, sps.coo_array if generation == "modern" else sps.coo_matrix)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["userId"].nunique()
    assert ncols == ml_ratings["movieId"].nunique()

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
    assert nrows == ml_ratings["userId"].nunique()
    assert ncols == ml_ratings["movieId"].nunique()

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
    assert nrows == ml_ratings["userId"].nunique()
    assert ncols == ml_ratings["movieId"].nunique()

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
    assert nrows == ml_ratings["userId"].nunique()
    assert ncols == ml_ratings["movieId"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.indptr)
    _check_item_number_counts(ml_ds, ml_ratings, log.indices)
    _check_item_ids(ml_ds, ml_ratings, log.indices)

    # right rating values
    assert np.all(log.data == 1.0)


@mark.parametrize("generation", ["modern", "legacy"])
def test_matrix_scipy_missing_rating(ml_ratings: pd.DataFrame, generation):
    ml_ds = from_interactions_df(ml_ratings[["userId", "movieId", "timestamp"]], item_col="movieId")
    log = ml_ds.interaction_matrix(format="scipy", field="rating", legacy=generation == "legacy")
    assert isinstance(log, sps.csr_array if generation == "modern" else sps.csr_matrix)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["userId"].nunique()
    assert ncols == ml_ratings["movieId"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.indptr)
    _check_item_number_counts(ml_ds, ml_ratings, log.indices)
    _check_item_ids(ml_ds, ml_ratings, log.indices)
    assert np.allclose(log.data, 1.0)


def test_matrix_torch_csr(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="torch", field="rating")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse_csr
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = log.shape
    assert nrows == ml_ratings["userId"].nunique()
    assert ncols == ml_ratings["movieId"].nunique()

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
    assert nrows == ml_ratings["userId"].nunique()
    assert ncols == ml_ratings["movieId"].nunique()

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
    assert nrows == ml_ratings["userId"].nunique()
    assert ncols == ml_ratings["movieId"].nunique()

    _check_user_number_counts(ml_ds, ml_ratings, log.indices()[0, :])
    _check_user_ids(ml_ds, ml_ratings, log.indices()[0, :])
    _check_item_number_counts(ml_ds, ml_ratings, log.indices()[1, :])
    _check_item_ids(ml_ds, ml_ratings, log.indices()[1, :])
    _check_ratings(ml_ds, ml_ratings, log.values().numpy())


def test_matrix_torch_missing_rating(ml_ratings: pd.DataFrame):
    ml_ds = from_interactions_df(ml_ratings[["userId", "movieId", "timestamp"]], item_col="movieId")
    log = ml_ds.interaction_matrix(format="torch", field="rating")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse_csr
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["userId"].nunique()
    assert ncols == ml_ratings["movieId"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.crow_indices())
    _check_item_number_counts(ml_ds, ml_ratings, log.col_indices())
    _check_item_ids(ml_ds, ml_ratings, log.col_indices())
    assert np.allclose(log.values().numpy(), 1.0)


def test_matrix_torch_timestamp(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="torch", field="timestamp")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse_csr
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = log.shape
    assert nrows == ml_ratings["userId"].nunique()
    assert ncols == ml_ratings["movieId"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.crow_indices())
    _check_item_number_counts(ml_ds, ml_ratings, log.col_indices())
    _check_item_ids(ml_ds, ml_ratings, log.col_indices())
    _check_timestamp(ml_ds, ml_ratings, log.values().numpy())
