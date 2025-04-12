# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Matrix layouts.
"""

# pyright: basic
from __future__ import annotations

from typing import NamedTuple, TypeVar

import numpy as np
import pyarrow as pa
import scipy.sparse as sps
import torch

t = torch
M = TypeVar("M", "CSRStructure", sps.csr_array, sps.coo_array, sps.spmatrix, t.Tensor)


class CSRStructure(NamedTuple):
    """
    Representation of the compressed sparse row structure of a sparse matrix,
    without any data values.

    Stability:
        Caller
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


class COOStructure(NamedTuple):
    """
    Representation of the coordinate structure of a sparse matrix, without any
    data values.

    Stability:
        Caller
    """

    row_numbers: np.ndarray
    col_numbers: np.ndarray
    shape: tuple[int, int]

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    @property
    def nnz(self):
        return self.row_numbers[self.nrows]


def sparse_to_arrow(arr: sps.csr_array) -> pa.ListArray:
    """
    Convert a spare matrix into a PyArrow list array.  The
    resulting array has 32-bit column indices and values.
    """

    cols = pa.array(arr.indices, pa.int32())
    vals = pa.array(arr.data, pa.float32())

    entries = pa.StructArray.from_arrays([cols, vals], ["index", "value"])
    rows = pa.ListArray.from_arrays(pa.array(arr.indptr, pa.int32()), entries)
    return rows


def sparse_from_arrow(arr: pa.ListArray, shape: tuple[int, int] | None = None) -> sps.csr_array:
    """
    Convert a spare matrix into a PyArrow list array.  The
    resulting array has 32-bit column indices and values.
    """
    entries = arr.values
    if not pa.types.is_struct(entries.type):
        raise TypeError(f"entry type {entries.type}, expected struct")
    assert isinstance(entries, pa.StructArray)

    cols = entries.field("index")
    vals = entries.field("value")

    return sps.csr_array((vals.to_numpy(), cols.to_numpy(), arr.offsets.to_numpy()), shape=shape)
