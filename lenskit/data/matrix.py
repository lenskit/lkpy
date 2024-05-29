# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data manipulation routines.
"""

# pyright: basic
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch as t
from csr import CSR
from typing_extensions import Any, Generic, Literal, NamedTuple, Optional, TypeVar, overload

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
    users: pd.Index[Any]
    items: pd.Index[Any]


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
    users: Optional[pd.Index[Any]] = None,
    items: Optional[pd.Index[Any]] = None,
) -> RatingMatrix[CSR]: ...
@overload
def sparse_ratings(
    ratings: pd.DataFrame,
    scipy: Literal[True] | Literal["csr"],
    *,
    users: Optional[pd.Index[Any]] = None,
    items: Optional[pd.Index[Any]] = None,
) -> RatingMatrix[sps.csr_matrix]: ...
@overload
def sparse_ratings(
    ratings: pd.DataFrame,
    scipy: Literal["coo"],
    *,
    users: Optional[pd.Index[Any]] = None,
    items: Optional[pd.Index[Any]] = None,
) -> RatingMatrix[sps.coo_matrix]: ...
@overload
def sparse_ratings(
    ratings: pd.DataFrame,
    *,
    torch: Literal[True],
    users: Optional[pd.Index[Any]] = None,
    items: Optional[pd.Index[Any]] = None,
) -> RatingMatrix[t.Tensor]: ...
def sparse_ratings(
    ratings: pd.DataFrame,
    scipy: bool | Literal["csr", "coo"] = False,
    *,
    torch: bool = False,
    users: Optional[pd.Index[Any]] = None,
    items: Optional[pd.Index[Any]] = None,
) -> RatingMatrix[Any]:
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
        case "center":
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


def _nsr_unit(matrix: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    sqmat = t.sparse_csr_tensor(
        crow_indices=matrix.crow_indices(),
        col_indices=matrix.col_indices(),
        values=matrix.values().square(),
    )
    norms = sqmat.sum(dim=1, keepdim=True).to_dense().reshape(matrix.shape[0])
    norms.sqrt_()
    recip_norms = t.where(norms > 0, t.reciprocal(norms), 0.0)
    return t.sparse_csr_tensor(
        crow_indices=matrix.crow_indices(),
        col_indices=matrix.col_indices(),
        values=matrix.values() * t.repeat_interleave(recip_norms, matrix.crow_indices().diff()),
        size=matrix.shape,
    ), norms


def torch_sparse_from_scipy(
    M: sps.coo_array, layout: Literal["csr", "coo", "csc"] = "coo"
) -> t.Tensor:
    """
    Convert a SciPy :class:`sps.coo_array` into a torch sparse tensor.
    """
    ris = t.from_numpy(M.row)
    cis = t.from_numpy(M.col)
    vs = t.from_numpy(M.data)
    indices = t.stack([ris, cis])
    T = t.sparse_coo_tensor(indices, vs, size=M.shape)

    match layout:
        case "csr":
            return T.to_sparse_csr()
        case "csc":
            return T.to_sparse_csc()
        case "coo":
            return T.coalesce()
        case _:
            raise ValueError(f"invalid layout {layout}")
