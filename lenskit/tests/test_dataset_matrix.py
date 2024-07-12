"""
Tests for the Dataset class.
"""

from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
from numpy.typing import ArrayLike

from pytest import raises

from lenskit.data import Dataset
from lenskit.data.dataset import FieldError, from_interactions_df
from lenskit.data.matrix import CSRStructure
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401


def _check_user_offset_counts(ml_ds: Dataset, ml_ratings: pd.DataFrame, offsets: ArrayLike):
    user_counts = ml_ratings["userId"].value_counts().reindex(ml_ds.user_vocab)
    row_lens = np.diff(offsets)
    assert np.all(row_lens == user_counts)


def _check_user_number_counts(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    users, counts = np.unique(nums, return_counts=True)
    user_counts = ml_ratings["userId"].value_counts().reindex(ml_ds.user_vocab[users])
    assert np.all(counts == user_counts)


def _check_item_number_counts(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    items, counts = np.unique(nums, return_counts=True)
    item_counts = ml_ratings["movieId"].value_counts().reindex(ml_ds.item_vocab[items])
    assert np.all(counts == item_counts)


def _check_user_ids(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    assert np.all(ml_ds.user_vocab[np.asarray(nums)] == ml_ratings["movieId"])


def _check_item_ids(ml_ds: Dataset, ml_ratings: pd.DataFrame, nums: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    assert np.all(ml_ds.item_vocab[np.asarray(nums)] == ml_ratings["movieId"])


def _check_ratings(ml_ds: Dataset, ml_ratings: pd.DataFrame, rates: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    assert np.all(rates == ml_ratings["rating"])


def _check_timestamp(ml_ds: Dataset, ml_ratings: pd.DataFrame, ts: ArrayLike):
    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    assert np.all(ts == ml_ratings["timestamp"])


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

    # right rating values
    assert np.all(log.data == 1.0)


def test_matrix_pandas_missing_rating(ml_ratings: pd.DataFrame):
    ml_ds = from_interactions_df(ml_ratings[["userId", "movieId", "timestamp"]], item_col="movieId")
    log = ml_ds.interaction_matrix(format="pandas", field="rating")
    assert isinstance(log, sps.csr_array)
    assert log.nnz == len(ml_ratings)

    _check_user_number_counts(ml_ds, ml_ratings, log["user_num"])
    _check_user_ids(ml_ds, ml_ratings, log["user_num"])
    _check_item_number_counts(ml_ds, ml_ratings, log.indices)
    _check_item_ids(ml_ds, ml_ratings, log.indices)
    assert np.all(log["rating"] == 1.0)


def test_matrix_scipy_coo(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="scipy", layout="coo", field="rating")
    assert isinstance(log, sps.coo_array)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user"].nunique()
    assert ncols == ml_ratings["item"].nunique()

    _check_user_number_counts(ml_ds, ml_ratings, log.coords[0])
    _check_user_ids(ml_ds, ml_ratings, log.coords[0])
    _check_item_number_counts(ml_ds, ml_ratings, log.coords[1])
    _check_item_ids(ml_ds, ml_ratings, log.coords[1])
    _check_ratings(ml_ds, ml_ratings, log.data)


def test_matrix_scipy_csr(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="scipy", field="rating")
    assert isinstance(log, sps.csr_array)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user"].nunique()
    assert ncols == ml_ratings["item"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.indptr)
    _check_item_number_counts(ml_ds, ml_ratings, log.indices)
    _check_item_ids(ml_ds, ml_ratings, log.indices)
    _check_ratings(ml_ds, ml_ratings, log.data)


def test_matrix_scipy_indicator(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="scipy")
    assert isinstance(log, sps.csr_array)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user"].nunique()
    assert ncols == ml_ratings["item"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.indptr)
    _check_item_number_counts(ml_ds, ml_ratings, log.indices)
    _check_item_ids(ml_ds, ml_ratings, log.indices)

    # right rating values
    assert np.all(log.data == 1.0)


def test_matrix_scipy_missing_rating(ml_ratings: pd.DataFrame):
    ml_ds = from_interactions_df(ml_ratings[["user", "item", "timestamp"]])
    log = ml_ds.interaction_matrix(format="scipy", field="rating")
    assert isinstance(log, sps.csr_array)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user"].nunique()
    assert ncols == ml_ratings["item"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.indptr)
    _check_item_number_counts(ml_ds, ml_ratings, log.indices)
    _check_item_ids(ml_ds, ml_ratings, log.indices)
    _check_ratings(ml_ds, ml_ratings, log.data)


def test_matrix_torch_csr(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="torch", field="rating")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse_csr
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = log.shape
    assert nrows == ml_ratings["user"].nunique()
    assert ncols == ml_ratings["item"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.crow_indices())
    _check_item_number_counts(ml_ds, ml_ratings, log.col_indices())
    _check_item_ids(ml_ds, ml_ratings, log.col_indices())
    _check_ratings(ml_ds, ml_ratings, log.values())


def test_matrix_torch_indicator(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="torch")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse_csr
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = log.shape
    assert nrows == ml_ratings["user"].nunique()
    assert ncols == ml_ratings["item"].nunique()

    _check_user_offset_counts(ml_ds, ml_ratings, log.crow_indices())
    _check_item_number_counts(ml_ds, ml_ratings, log.col_indices())
    _check_item_ids(ml_ds, ml_ratings, log.col_indices())
    assert np.all(log.values() == 1.0)


def test_matrix_torch_coo(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="torch", layout="coo", field="rating")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user"].nunique()
    assert ncols == ml_ratings["item"].nunique()

    _check_user_number_counts(ml_ds, ml_ratings, log.row_indices())
    _check_user_ids(ml_ds, ml_ratings, log.row_indices())
    _check_item_number_counts(ml_ds, ml_ratings, log.col_indices())
    _check_item_ids(ml_ds, ml_ratings, log.col_indices())
    _check_ratings(ml_ds, ml_ratings, log.values())
