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
import warnings
from typing import Any, NamedTuple, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sps
import torch
from numpy.typing import ArrayLike

t = torch
M = TypeVar("M", "CSRStructure", sps.csr_array, sps.coo_array, sps.spmatrix, t.Tensor)

SPARSE_IDX_EXT_NAME = "lenskit.sparse_index"
SPARSE_IDX_LIST_EXT_NAME = "lenskit.sparse_index_list"
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


class SparseIndexType(pa.ExtensionType):
    """
    Data type for the index field of a sparse row.  Indexes are just stored as
    ``int32``s; the extension type attaches the row's dimensionality to the index
    field (making it easier to pass it to/from Rust, since we often pass arrays
    and not entire fields).

    .. stability:: internal
    """

    dimension: int

    def __init__(self, dimension: int):
        self.dimension = dimension
        super().__init__(pa.int32(), SPARSE_IDX_EXT_NAME)

    def check_dimension(self, expected: int | None) -> int:
        """
        Check that this index type has the expected dimension.

        Returns:
            The dimension of the index type.

        Raises:
            ValueError:
                If the type's dimension does not match the expected dimension.
        """
        if expected is not None and expected != self.dimension:
            raise ValueError(f"dimension mismatch: expected {expected}, found {self.dimension}")
        return self.dimension

    def __arrow_ext_serialize__(self) -> bytes:
        return json.dumps({"dimension": self.dimension}).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        data = json.loads(serialized.decode())
        if not pa.types.is_int32(storage_type):
            raise TypeError("sparse index must be int32")
        return cls(data["dimension"])


class SparseIndexListType(pa.ExtensionType):
    """
    Sparse index lists.  These are the row type for structure-only sparse
    matrices.
    """

    value_type: None = None
    index_type: SparseIndexType

    def __init__(self, dimension: int, large: bool = False):
        self.index_type = SparseIndexType(dimension)

        lt_ctor = pa.large_list if large else pa.list_

        super().__init__(
            lt_ctor(self.index_type),
            SPARSE_IDX_LIST_EXT_NAME,
        )

    @classmethod
    def from_type(cls, data_type: pa.DataType, dimension: int | None = None) -> SparseIndexListType:
        """
        Create a sparse index list type from an Arrow data type, handling legacy struct
        layouts without the extension types.

        Args:
            data_type:
                The Arrow data type to interpret as a row type.
            dimension:
                The row dimension, if known from an external source.  If
                provided and the data type also includes the dimensionality,
                both dimensions must match.

        Raises:
            TypeError:
                If the data type is not a valid sparse row type.
            ValueError:
                If there is another error, such as mismatched dimensions.
        """
        if isinstance(data_type, SparseIndexListType):
            data_type.index_type.check_dimension(dimension)
            return data_type

        if pa.types.is_list(data_type):
            large = False
        elif pa.types.is_large_list(data_type):
            large = True
        else:
            raise TypeError(f"expected list type, found {data_type}")
        inner = data_type.value_type  # type: ignore

        dimension = _check_index_type(inner, dimension)

        return cls(dimension, large=large)

    @property
    def dimension(self) -> int:
        return self.index_type.dimension

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls.from_type(storage_type)

    def __arrow_ext_class__(self):
        return SparseRowArray


class SparseRowType(pa.ExtensionType):
    """
    Data type for sparse rows stored in Arrow.  Sparse rows are stored as lists
    of structs with ``index`` and ``column`` fields.

    .. stability:: internal
    """

    value_type: pa.DataType | None
    index_type: SparseIndexType

    def __init__(
        self, dimension: int, value_type: pa.DataType | None = pa.float32(), large: bool = False
    ):
        self.value_type = value_type
        self.index_type = SparseIndexType(dimension)

        lt_ctor = pa.large_list if large else pa.list_

        if value_type is None:
            element_type = self.index_type
        else:
            element_type = pa.struct(
                [
                    ("index", self.index_type),
                    ("value", value_type),
                ]
            )

        super().__init__(
            lt_ctor(element_type),
            SPARSE_ROW_EXT_NAME,
        )

    @classmethod
    def from_type(cls, data_type: pa.DataType, dimension: int | None = None) -> SparseRowType:
        """
        Create a sparse row type from an Arrow data type, handling legacy struct
        layouts without the extension types.

        Args:
            data_type:
                The Arrow data type to interpret as a row type.
            dimension:
                The row dimension, if known from an external source.  If
                provided and the data type also includes the dimensionality,
                both dimensions must match.

        Raises:
            TypeError:
                If the data type is not a valid sparse row type.
            ValueError:
                If there is another error, such as mismatched dimensions.
        """
        if isinstance(data_type, SparseRowType):
            data_type.index_type.check_dimension(dimension)
            return data_type

        if pa.types.is_list(data_type):
            large = False
        elif pa.types.is_large_list(data_type):
            large = True
        else:
            raise TypeError(f"expected list type, found {data_type}")
        inner = data_type.value_type  # type: ignore

        if not pa.types.is_struct(inner):
            raise TypeError(f"expected struct type, found {inner}")

        if inner.num_fields != 2:
            raise TypeError(f"element struct must have 2 elements, found {inner.num_fields}")

        idx_f = inner.field(0)
        assert isinstance(idx_f, pa.Field)
        if idx_f.name != "index":
            raise TypeError(f"first field of element struct must be 'index', found {idx_f.name}")

        dimension = _check_index_type(idx_f.type, dimension)

        val_f = inner.field(1)
        if val_f.name != "value":
            raise TypeError(f"second field of element struct must be 'value', found {val_f.name}")

        return cls(dimension, val_f.type, large=large)

    @property
    def dimension(self) -> int:
        return self.index_type.dimension

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls.from_type(storage_type)

    def __arrow_ext_class__(self):
        return SparseRowArray


class SparseRowArray(pa.ExtensionArray):
    """
    An array of sparse rows (a compressed sparse row matrix).

    .. stability:: internal
    """

    type: SparseRowType | SparseIndexListType

    @classmethod
    def from_arrays(
        cls,
        offsets: ArrayLike,
        indices: ArrayLike,
        values: ArrayLike | None = None,
        *,
        shape: tuple[int, int] | None = None,
    ) -> SparseRowArray:
        offsets = pa.array(offsets)
        large = pa.types.is_int64(offsets.type)
        indices = pa.array(indices, type=pa.int32())

        if shape:
            _nr, nc = shape
        else:
            nc = pc.max(indices).to_py() + 1

        if values is not None:
            values = pa.array(values)
            row_type = SparseRowType(nc, values.type, large)
            index_type = row_type.index_type

            elements = pa.StructArray.from_arrays(
                [indices.cast(index_type), values], names=["index", "value"]
            )
        else:
            row_type = SparseIndexListType(nc, large)
            elements = indices.cast(row_type.index_type)

        if large:
            storage = pa.LargeListArray.from_arrays(offsets, elements)
        else:
            storage = pa.ListArray.from_arrays(offsets, elements)

        return pa.ExtensionArray.from_storage(row_type, storage)  # type: ignore

    @classmethod
    def from_array(cls, array: pa.Array, dimension: int | None = None) -> SparseRowArray:
        """
        Interpret an Arrow array as a sparse row array, if possible.  Handles
        legacy layouts without the extension types.

        Args:
            array:
                The array to convert.
            dimension:
                The dimensionality of the sparse rows, if known from an external source.
        """
        if isinstance(array, SparseRowArray):
            assert isinstance(array.type, SparseRowType)
            array.type.index_type.check_dimension(dimension)
            return array
        elif isinstance(array.type, SparseRowType):
            array.type.index_type.check_dimension(dimension)
            return pa.ExtensionArray.from_storage(array.type, array)  # type: ignore

        et = SparseRowType.from_type(array.type, dimension)
        return pa.ExtensionArray.from_storage(et, array.cast(et.storage_type))  # type: ignore

    @classmethod
    def from_scipy(
        cls, csr: sps.csr_array[Any, tuple[int, int]], *, large: bool | None = None
    ) -> SparseRowArray:
        _nr, dim = csr.shape
        smax = np.iinfo(np.int32).max

        offsets = csr.indptr
        if large:
            offsets = np.require(offsets, np.int64)
        elif csr.nnz < smax:
            offsets = np.require(offsets, dtype=np.int32)
        elif large is False:
            raise ValueError("sparse matrix size {:,d} too large for list".format(csr.nnz))
        cols = pa.array(csr.indices, SparseIndexType(dim))
        vals = pa.array(csr.data)

        return cls.from_arrays(offsets, cols, vals, shape=csr.shape)

    def to_scipy(self) -> sps.csr_array[Any, tuple[int, int]]:
        """
        Convert this sparse row array to a SciPy sparse array.
        """
        if not self.has_values:
            raise TypeError("structure-only arrays cannot convert to scipy")
        return sps.csr_array(
            (self.values.to_numpy(), self.indices.to_numpy(), self.offsets.to_numpy()),
            shape=(len(self), self.type.dimension),
        )

    def to_torch(self) -> torch.Tensor:
        """
        Convert this sparse row array to a Torch sparse tensor.
        """
        if not self.has_values:
            raise TypeError("structure-only arrays cannot convert to Torch")
        return torch.sparse_csr_tensor(
            crow_indices=torch.as_tensor(
                self.offsets.to_numpy(zero_copy_only=False, writable=True)
            ),
            col_indices=torch.as_tensor(self.indices.to_numpy(zero_copy_only=False, writable=True)),
            values=torch.as_tensor(self.values.to_numpy(zero_copy_only=False, writable=True)),
            size=self.shape,
        )

    @property
    def dimension(self) -> int:
        """
        Get the number of columns in the sparse matrix.
        """
        return self.type.dimension

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), self.dimension)

    @property
    def has_values(self) -> bool:
        return self.type.value_type is not None

    @property
    def offsets(self) -> pa.Int32Array:
        return self.storage.offsets

    @property
    def indices(self) -> pa.Int32Array:
        if self.has_values:
            return self.storage.values.field(0)
        else:
            return self.storage.values

    @property
    def values(self) -> pa.Array | None:
        if self.has_values:
            return self.storage.values.field(1)
        else:
            return None

    def row_extent(self, row: int) -> tuple[int, int]:
        """
        Get the start and end of a row.
        """
        start = self.storage.offsets[row].as_py()
        end = self.storage.offsets[row + 1].as_py()
        return start, end

    def row_indices(self, row: int) -> pa.Int32Array:
        """
        Get the index array for a compressed sparse row.
        """
        sp, ep = self.row_extent(row)
        indices = self.indices
        return indices.slice(sp, ep - sp)


def _check_index_type(index_type: pa.DataType, dimension: int | None) -> int:
    if isinstance(index_type, SparseIndexType):
        return index_type.check_dimension(dimension)
    elif dimension is None:
        raise TypeError(
            f"index type must be SparseIndex when no external dimension, found {index_type}"
        )
    elif not pa.types.is_int32(index_type):
        if pa.types.is_int64(index_type):
            warnings.warn("sparse row has legacy int64 indices", DeprecationWarning)
        else:
            raise TypeError(f"index type must be SparseIndex or int32, found {index_type}")

    return dimension


pa.register_extension_type(SparseIndexType(0))  # type: ignore
pa.register_extension_type(SparseIndexListType(0))  # type: ignore
pa.register_extension_type(SparseRowType(0))  # type: ignore
