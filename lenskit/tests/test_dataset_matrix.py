"""
Tests for the Dataset class.
"""

from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch

from lenskit.data import Dataset
from lenskit.data.matrix import CSRStructure
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401


def test_matrix_structure(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="structure")
    assert isinstance(log, CSRStructure)
    assert log.nnz == len(ml_ratings)

    assert log.nrows == ml_ratings["user"].nunique()
    assert log.ncols == ml_ratings["item"].nunique()

    # make sure we have the right # of ratings per user
    user_counts = ml_ratings["user"].value_counts().reindex(ml_ds.user_vocab)
    row_lens = np.diff(log.rowptrs)
    assert np.all(row_lens == user_counts)

    # right # of ratings per item
    items, counts = np.unique(log.colinds, return_counts=True)
    item_counts = ml_ratings["item"].value_counts().reindex(ml_ds.item_vocab[items])
    assert np.all(counts == item_counts)

    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    # right item IDs
    assert np.all(ml_ds.item_vocab[log.colinds] == ml_ratings["movieId"])


def test_matrix_scipy_coo(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="scipy", layout="coo")
    assert isinstance(log, sps.coo_array)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user"].nunique()
    assert ncols == ml_ratings["item"].nunique()

    # make sure we have the right # of ratings per user
    users, counts = np.unique(log.coords[0], return_counts=True)
    user_counts = ml_ratings["user"].value_counts().reindex(ml_ds.user_vocab[users])
    assert np.all(counts == user_counts)

    # right # of ratings per item
    items, counts = np.unique(log.coords[1], return_counts=True)
    item_counts = ml_ratings["item"].value_counts().reindex(ml_ds.item_vocab[items])
    assert np.all(counts == item_counts)

    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    # right item and user IDs
    assert np.all(ml_ds.item_vocab[log.coords[0]] == ml_ratings["userId"])
    assert np.all(ml_ds.item_vocab[log.coords[1]] == ml_ratings["movieId"])

    # right rating values
    assert np.all(log.data == ml_ratings["rating"])


def test_matrix_scipy_csr(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="scipy")
    assert isinstance(log, sps.csr_array)
    assert log.nnz == len(ml_ratings)

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user"].nunique()
    assert ncols == ml_ratings["item"].nunique()

    # make sure we have the right # of ratings per user
    user_counts = ml_ratings["user"].value_counts().reindex(ml_ds.user_vocab)
    row_lens = np.diff(log.indptr)
    assert np.all(row_lens == user_counts)

    # right # of ratings per item
    items, counts = np.unique(log.indices, return_counts=True)
    item_counts = ml_ratings["item"].value_counts().reindex(ml_ds.item_vocab[items])
    assert np.all(counts == item_counts)

    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    # right item IDs
    assert np.all(ml_ds.item_vocab[log.indices] == ml_ratings["movieId"])

    # right rating values
    assert np.all(log.data == ml_ratings["rating"])


def test_matrix_torch_csr(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="torch")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse_csr
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = log.shape
    assert nrows == ml_ratings["user"].nunique()
    assert ncols == ml_ratings["item"].nunique()

    # make sure we have the right # of ratings per user
    user_counts = ml_ratings["user"].value_counts().reindex(ml_ds.user_vocab)
    row_lens = np.diff(log.crow_indices())
    assert np.all(row_lens == user_counts)

    # right # of ratings per item
    items, counts = np.unique(log.col_indices(), return_counts=True)
    item_counts = ml_ratings["item"].value_counts().reindex(ml_ds.item_vocab[items])
    assert np.all(counts == item_counts)

    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    # right item IDs
    assert np.all(ml_ds.item_vocab[log.col_indices().numpy()] == ml_ratings["movieId"])

    # right rating values
    assert np.all(log.values() == ml_ratings["rating"])


def test_matrix_torch_coo(ml_ratings: pd.DataFrame, ml_ds: Dataset):
    log = ml_ds.interaction_matrix(format="torch", layout="coo")
    assert isinstance(log, torch.Tensor)
    assert log.is_sparse
    assert log.values().shape == torch.Size([len(ml_ratings)])

    nrows, ncols = cast(tuple[int, int], log.shape)
    assert nrows == ml_ratings["user"].nunique()
    assert ncols == ml_ratings["item"].nunique()

    # make sure we have the right # of ratings per user
    users, counts = np.unique(log.row_indices(), return_counts=True)
    user_counts = ml_ratings["user"].value_counts().reindex(ml_ds.user_vocab[users])
    assert np.all(counts == user_counts)

    # right # of ratings per item
    items, counts = np.unique(log.col_indices(), return_counts=True)
    item_counts = ml_ratings["item"].value_counts().reindex(ml_ds.item_vocab[items])
    assert np.all(counts == item_counts)

    ml_ratings = ml_ratings.sort_values(["userId", "movieId"])
    # right item and user IDs
    assert np.all(ml_ds.item_vocab[log.row_indices().numpy()] == ml_ratings["userId"])
    assert np.all(ml_ds.item_vocab[log.col_indices().numpy()] == ml_ratings["movieId"])

    # right rating values
    assert np.all(log.values() == ml_ratings["rating"])
