# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data manipulation routines.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch as t
from csr import CSR
from typing_extensions import Generic, Literal, NamedTuple, TypeVar, overload

_log = logging.getLogger(__name__)

M = TypeVar("M", CSR, sps.csr_matrix, sps.coo_matrix, t.Tensor)


class RatingMatrix(NamedTuple, Generic[M]):
    """
    A rating matrix with associated indices.

    Attributes:
        matrix:
            The rating matrix, with users on rows and items on columns.
        users: mapping from user IDs to row numbers.
        items: mapping from item IDs to column numbers.
    """

    matrix: M
    users: pd.Index
    items: pd.Index


class DimStats(NamedTuple):
    """
    The statistics for a matrix along a dimension (e.g. rows or columns).
    """

    "The size along this dimension."
    n: int
    "The other dimension of the matrix."
    n_other: int
    "The number of stored entries for each element."
    counts: t.Tensor
    "The sum of entries for each element."
    sums: t.Tensor
    "The mean of stored entries for each element."
    means: t.Tensor


@overload
def sparse_ratings(
    ratings: pd.DataFrame,
    *,
    users=None,
    items=None,
) -> RatingMatrix[CSR]: ...
@overload
def sparse_ratings(
    ratings: pd.DataFrame,
    scipy: Literal[True] | Literal["csr"],
    *,
    users=None,
    items=None,
) -> RatingMatrix[sps.csr_matrix]: ...
@overload
def sparse_ratings(
    ratings: pd.DataFrame,
    scipy: Literal["coo"],
    *,
    users=None,
    items=None,
) -> RatingMatrix[sps.coo_matrix]: ...
@overload
def sparse_ratings(
    ratings: pd.DataFrame,
    *,
    torch: Literal[True],
    users=None,
    items=None,
) -> RatingMatrix[t.Tensor]: ...
def sparse_ratings(
    ratings: pd.DataFrame,
    scipy: bool | Literal["csr", "coo"] = False,
    *,
    torch: bool = False,
    users=None,
    items=None,
):
    """
    Convert a rating table to a sparse matrix of ratings.

    Args:
        ratings: a data table of (user, item, rating) triples.
        scipy:
            if ``True`` or ``'csr'``, return a SciPy csr matrix instead of
            :class:`CSR`. if ``'coo'``, return a SciPy coo matrix.
        torch:
            if ``True``, return a PyTorch sparse tensor instead of a :class:`CSR`.
        users: an index of user IDs.
        items: an index of items IDs.

    Returns:
        RatingMatrix:
            a named tuple containing the sparse matrix, user index, and item index.
    """
    if users is None:
        users = pd.Index(np.unique(ratings.user), name="user")

    if items is None:
        items = pd.Index(np.unique(ratings.item), name="item")

    n = len(ratings)
    ni = len(items)
    nu = len(users)

    _log.debug("creating matrix with %d ratings for %d items by %d users", n, ni, nu)

    row_ind = users.get_indexer(ratings.user).astype(np.intc)
    if np.any(row_ind < 0):
        raise ValueError("provided user index does not cover all users")
    col_ind = items.get_indexer(ratings.item).astype(np.intc)
    if np.any(col_ind < 0):
        raise ValueError("provided item index does not cover all users")

    if "rating" in ratings.columns:
        vals = np.require(ratings.rating.values, np.float64)
    else:
        vals = None

    if scipy == "coo":
        matrix = sps.coo_matrix((vals, (row_ind, col_ind)), shape=(nu, ni))
    elif torch:
        if vals is None:
            vals = t.ones((len(ratings),), dtype=t.float32)
        else:
            vals = t.from_numpy(vals).to(t.float32)
        indices = t.stack([t.from_numpy(row_ind), t.from_numpy(col_ind)], dim=0)
        matrix = t.sparse_coo_tensor(indices, vals, size=(nu, ni))
        matrix = matrix.to_sparse_csr()
    else:
        matrix = CSR.from_coo(row_ind, col_ind, vals, (len(users), len(items)))
        if scipy:
            matrix = matrix.to_scipy()

    return RatingMatrix(matrix, users, items)  # pyright: ignore


def sparse_row_stats(matrix: t.Tensor) -> DimStats:
    if not matrix.is_sparse_csr:
        raise TypeError("only sparse CSR matrice supported")

    n, n_other = matrix.shape
    counts = matrix.crow_indices().diff()
    assert counts.shape == (n,), f"count shape {counts.shape} != {n}"
    sums = matrix.sum(dim=1, keepdim=True).to_dense().reshape(n)
    assert sums.shape == (n,), f"sum shape {sums.shape} != {n}"
    means = sums / counts

    return DimStats(n, n_other, counts, sums, means)


@overload
def normalize_sparse_rows(
    matrix: t.Tensor, method: Literal["center"], inplace: bool = False
) -> tuple[t.Tensor, t.Tensor]: ...
@overload
def normalize_sparse_rows(
    matrix: t.Tensor, method: Literal["unit"], inplace: bool = False
) -> tuple[t.Tensor, t.Tensor]: ...
def normalize_sparse_rows(
    matrix: t.Tensor, method: str, inplace: bool = False
) -> tuple[t.Tensor, t.Tensor]:
    """
    Normalize the rows of a sparse matrix.
    """
    match method:
        case "unit":
            return _nsr_unit(matrix)
        case "mean":
            return _nsr_mean_center(matrix)
        case _:
            raise ValueError(f"unsupported normalization method {method}")


def _nsr_mean_center(matrix: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    stats = sparse_row_stats(matrix)
    return t.sparse_csr_tensor(
        crow_indices=matrix.crow_indices(),
        col_indices=matrix.col_indices(),
        values=matrix.values() - t.repeat_interleave(stats.means, stats.counts),
        size=matrix.shape,
    ), stats.means
