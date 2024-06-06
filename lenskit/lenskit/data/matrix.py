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
import platform

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
from typing_extensions import Any, Generic, Literal, NamedTuple, Optional, TypeVar, overload

_log = logging.getLogger(__name__)

t = torch
M = TypeVar("M", sps.csr_array, sps.coo_array, t.Tensor)


class CSRStructure(NamedTuple):
    """
    Representation of the compressed sparse row structure of a sparse matrix,
    without any data values.
    """

    rowptrs: np.ndarray
    colinds: np.ndarray
    shape: tuple[int, int]

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    @property
    def nnz(self):
        return self.rowptrs[self.nrows]


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
    type: Literal["scipy"] = "scipy",
    layout: Literal["csr"] = "csr",
    users: Optional[pd.Index[Any]] = None,
    items: Optional[pd.Index[Any]] = None,
) -> RatingMatrix[sps.csr_array]: ...
@overload
def sparse_ratings(
    ratings: pd.DataFrame,
    *,
    type: Literal["scipy"] = "scipy",
    layout: Literal["coo"] = "coo",
    users: Optional[pd.Index[Any]] = None,
    items: Optional[pd.Index[Any]] = None,
) -> RatingMatrix[sps.coo_array]: ...
@overload
def sparse_ratings(
    ratings: pd.DataFrame,
    *,
    type: Literal["torch"],
    layout: Literal["coo", "csr"] = "csr",
    users: Optional[pd.Index[Any]] = None,
    items: Optional[pd.Index[Any]] = None,
) -> RatingMatrix[t.Tensor]: ...
def sparse_ratings(
    ratings: pd.DataFrame,
    *,
    type: Literal["scipy", "torch"] = "scipy",
    layout: Literal["csr", "coo"] = "csr",
    users: Optional[pd.Index[Any]] = None,
    items: Optional[pd.Index[Any]] = None,
) -> RatingMatrix[Any]:
    """
    Convert a rating table to a sparse matrix of ratings.

    Args:
        ratings:
            A data table of (user, item, rating) triples.
        type:
            The type of matrix to create.  Can be ``scipy`` to create a SciPy
            sparse array (see :mod:`scipy.sparse`), or ``torch`` to create a
            sparse tensor (see :mod:`torch.sparse`).
        layout:
            The matrix layout to use.
        users:
            An index of user IDs.
        items:
            An index of items IDs.

    Returns:
        RatingMatrix:
            a named tuple containing the sparse matrix, user index, and item
            index.
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

    if type == "torch":
        if "rating" in ratings.columns:
            vals = t.from_numpy(ratings["rating"].values).to(t.float32)
        else:
            vals = t.ones((len(ratings),), dtype=t.float32)
        indices = t.stack([t.from_numpy(row_ind), t.from_numpy(col_ind)], dim=0)
        matrix = t.sparse_coo_tensor(indices, vals, size=(nu, ni))
        if layout == "csr":
            matrix = matrix.to_sparse_csr()
    elif type == "scipy":
        if "rating" in ratings.columns:
            vals = ratings["rating"].values
        else:
            vals = np.ones((len(ratings),), dtype=np.float32)
        matrix = sps.coo_array((vals, (row_ind, col_ind)), shape=(nu, ni))
        if layout == "csr":
            matrix = matrix.tocsr()
    else:
        raise ValueError(f"unknown type {type}")

    return RatingMatrix(matrix, users, items)


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
    assert indices.shape == (2, M.nnz)
    T = t.sparse_coo_tensor(indices, vs, size=M.shape)
    assert T.shape == M.shape

    match layout:
        case "csr":
            return T.to_sparse_csr()
        case "csc":
            return T.to_sparse_csc()
        case "coo":
            return T.coalesce()
        case _:
            raise ValueError(f"invalid layout {layout}")


if platform.machine() == "arm64":

    @torch.jit.ignore  # type: ignore
    def safe_spmv(matrix, vector):  # type: ignore
        """
        Sparse matrix-vector multiplication working around PyTorch bugs.

        This is equivalent to :func:`torch.mv` for sparse CSR matrix
        and dense vector, but it works around PyTorch bug 127491_ by
        falling back to SciPy on ARM.

        .. _127491: https://github.com/pytorch/pytorch/issues/127491
        """
        assert matrix.is_sparse_csr
        nr, nc = matrix.shape
        M = sps.csr_array(
            (matrix.values().numpy(), matrix.col_indices().numpy(), matrix.crow_indices().numpy()),
            (nr, nc),
        )
        v = vector.numpy()
        return torch.from_numpy(M @ v)

else:

    def safe_spmv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """
        Sparse matrix-vector multiplication working around PyTorch bugs.

        This is equivalent to :func:`torch.mv` for sparse CSR matrix
        and dense vector, but it works around PyTorch bug 127491_ by
        falling back to SciPy on ARM.

        .. _127491: https://github.com/pytorch/pytorch/issues/127491
        """
        assert matrix.is_sparse_csr
        return torch.mv(matrix, vector)
