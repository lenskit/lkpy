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
import scipy.sparse as sps
import torch
from numpy.typing import ArrayLike
from typing_extensions import NamedTuple, Optional, TypeVar

_log = logging.getLogger(__name__)

t = torch
M = TypeVar("M", "CSRStructure", sps.csr_array, sps.coo_array, sps.spmatrix, t.Tensor)


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

    def extent(self, row: int) -> tuple[int, int]:
        return self.rowptrs[row], self.rowptrs[row + 1]

    def row_cs(self, row: int) -> np.ndarray:
        sp, ep = self.extent(row)
        return self.colinds[sp:ep]


class InteractionMatrix:
    """
    Internal helper class used by :class:`lenskit.data.Dataset` to store the
    user-item interaction matrix.  The data is stored simultaneously in CSR and
    COO format.  Most code has no need to interact with this class directly —
    :class:`~lenskit.data.Dataset` methods provide data in a range of formats.
    """

    n_obs: int
    n_users: int
    n_items: int

    user_nums: np.ndarray[int, np.dtype[np.int32]]
    "User (row) numbers."
    user_ptrs: np.ndarray[int, np.dtype[np.int32]]
    "User (row) offsets / pointers."
    item_nums: np.ndarray[int, np.dtype[np.int32]]
    "Item (column) numbers."
    ratings: Optional[np.ndarray[int, np.dtype[np.float32]]] = None
    "Rating values."
    timestamps: Optional[np.ndarray[int, np.dtype[np.int64]]] = None
    "Timestamps as 64-bit Unix timestamps."

    def __init__(
        self,
        users: ArrayLike,
        items: ArrayLike,
        ratings: Optional[ArrayLike],
        timestamps: Optional[ArrayLike],
        n_users: int,
        n_items: int,
    ):
        self.user_nums = np.asarray(users, np.int32)
        assert np.all(np.diff(self.user_nums) >= 0), "matrix data not sorted"
        self.item_nums = np.asarray(items, np.int32)
        if ratings is not None:
            self.ratings = np.asarray(ratings, np.float32)
        if timestamps is not None:
            self.timestamps = np.asarray(timestamps, np.int64)

        self.n_obs = len(self.user_nums)
        self.n_items = n_items
        self.n_users = n_users
        cp1 = np.zeros(self.n_users + 1, np.int32)
        np.add.at(cp1[1:], self.user_nums, 1)
        self.user_ptrs = cp1.cumsum(dtype=np.int32)
        if self.user_ptrs[-1] != len(self.user_nums):
            raise ValueError("mismatched counts and array sizes")

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the interaction matrix (rows x columns).
        """
        return (self.n_users, self.n_items)
