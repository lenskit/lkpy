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
from typing import Any, NamedTuple, TypeVar

import numpy as np
import pyarrow as pa
import scipy.sparse as sps
import torch

t = torch
M = TypeVar("M", "CSRStructure", sps.csr_array, sps.coo_array, sps.spmatrix, t.Tensor)

SPARSE_ROW_EXT_NAME = "lenskit.sparse_row"


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
            pa.list_(self.element_type(dimension, value_type)),
            SPARSE_ROW_EXT_NAME,
        )
        self.dimension = dimension
        self.value_type = value_type

    @classmethod
    def element_type(cls, dimension: int, value_type: pa.DataType = pa.float32()) -> pa.StructType:
        return pa.struct(
            [
                pa.field(
                    "index",
                    pa.int32(),
                    nullable=False,
                    metadata={"dimension": str(dimension)},
                ),
                ("value", value_type),
            ]
        )

    def __arrow_ext_serialize__(self) -> bytes:
        if hasattr(self, "dimension"):
            return json.dumps({"dimension": self.dimension}).encode()
        else:
            return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        if serialized:
            data = json.loads(serialized.decode())
        else:
            data = {}
        dim = data.get("dimension", None)

        if not (
            pa.types.is_list(storage_type)
            or pa.types.is_list_view(storage_type)
            or pa.types.is_large_list(storage_type)
            or pa.types.is_large_list_view(storage_type)
        ):
            raise TypeError(f"expected list type, found {storage_type}")
        inner = storage_type.value_type  # type: ignore
        if not pa.types.is_struct(inner):
            raise TypeError(f"expected struct element type, found {inner}")
        if len(inner.fields) != 2:
            raise TypeError(f"element struct must have 2 elements, found {len(inner.fields)}")
        idx_f = inner.fields[0]
        assert isinstance(idx_f, pa.Field)
        if idx_f.name != "index":
            raise TypeError(f"first field of element struct must be 'index', found {idx_f.name}")
        if idx_f.type != pa.int32():
            raise TypeError(f"index type must be Int32, found {idx_f.type}")
        if idx_f.metadata and b"dimension" in idx_f.metadata:
            i_dim = int(idx_f.metadata[b"dimension"])
            if dim is not None and i_dim != dim:
                raise ValueError(f"dimension mismatch: {i_dim} != {dim}")
            dim = i_dim

        val_f = inner.fields[1]
        if val_f.name != "value":
            raise TypeError(f"second field of element struct must be 'value', found {val_f.name}")

        return cls(data["dimension"], inner[1].type)

    def __arrow_ext_class__(self):
        return SparseRowArray


class SparseRowArray(pa.ExtensionArray):
    """
    An array of sparse rows (a compressed sparse row matrix).
    """

    @classmethod
    def from_csr(cls, csr: sps.csr_array[Any, tuple[int, int]]) -> SparseRowArray:
        offsets = pa.array(csr.indptr, pa.int32())
        cols = pa.array(csr.indices, pa.int32())
        vals = pa.array(csr.data)

        _nr, dim = csr.shape
        entries = pa.StructArray.from_arrays(
            [cols, vals], fields=SparseRowType.element_type(dim, vals.type).fields
        )
        rows = pa.ListArray.from_arrays(offsets, entries)
        return pa.ExtensionArray.from_storage(SparseRowType(csr.shape[1], vals.type), rows)  # type: ignore

    @property
    def column_count(self) -> int:
        """
        Get the number of columns in the sparse matrix.
        """
        assert isinstance(self.type, SparseRowType)
        return self.type.dimension

    @property
    def offsets(self) -> pa.Int32Array:
        return self.storage.offsets

    @property
    def indices(self) -> pa.Int32Array:
        return self.storage.values.field(0)

    @property
    def values(self) -> pa.Array:
        return self.storage.values.field(1)


pa.register_extension_type(SparseRowType(0))  # type: ignore


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
