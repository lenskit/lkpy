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

import json
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


class SparseRowType(pa.ExtensionType):
    """
    Data type for sparse rows stored in Arrow.  Sparse rows are stored as lists
    of structs with ``index`` and ``column`` fields.
    """

    value_type: pa.DataType
    dimension: int

    def __init__(self, dimension: int, value_type: pa.DataType = pa.float32()):
        super().__init__(
            pa.list_(pa.struct([("index", pa.int32()), ("value", value_type)])),
            "lenskit.sparse_row",
        )
        self.dimension = dimension
        self.value_type = value_type

    def __arrow_ext_serialize__(self) -> bytes:
        return json.dumps({"dimension": self.dimension}).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        data = json.loads(serialized.decode())
        assert (
            pa.types.is_list(storage_type)
            or pa.types.is_list_view(storage_type)
            or pa.types.is_large_list(storage_type)
            or pa.types.is_large_list_view(storage_type)
        )
        inner = storage_type.value_type
        assert pa.types.is_struct(inner)
        assert len(inner.fields) == 2
        return cls(data["dimension"], inner[1].type)

    def __arrow_ext_class__(self):
        return SparseRowArray


class SparseRowArray(pa.ExtensionArray):
    """
    An array of sparse rows (a compressed sparse row matrix).
    """

    @property
    def column_count(self) -> int:
        """
        Get the number of columns in the sparse matrix.
        """
        return self.type.dimension

    @property
    def indices(self) -> pa.Int32Array:
        return self.storage.field(0)

    @property
    def values(self) -> pa.Array:
        return self.storage.field(1)


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


def sparse_from_arrow(
    arr: pa.ListArray | pa.LargeListArray, shape: tuple[int, int] | None = None
) -> sps.csr_array:
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
