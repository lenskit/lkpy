# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data attribute accessors.
"""

from __future__ import annotations

import json
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_array
from typing_extensions import Any

from lenskit.data.matrix import SparseRowArray
from lenskit.torch import safe_tensor

from .schema import AttrLayout, ColumnSpec
from .types import IDArray
from .vocab import Vocabulary


def attr_set(
    name: str, spec: ColumnSpec, table: pa.Table, vocab: Vocabulary, rows: pa.Int32Array | None
):
    match spec.layout:
        case AttrLayout.SCALAR:
            return ScalarAttributeSet(name, spec, table, vocab, rows)
        case AttrLayout.LIST:
            return ListAttributeSet(name, spec, table, vocab, rows)
        case AttrLayout.VECTOR:
            return VectorAttributeSet(name, spec, table, vocab, rows)
        case AttrLayout.SPARSE:
            return SparseAttributeSet(name, spec, table, vocab, rows)
        case _:  # pragma: nocover
            raise ValueError(f"unsupported layout {spec.layout}")


class AttributeSet:
    """
    Base class for attributes associated with entities.

    This is the general interface for attribute sets.  Not all access methods
    are supported for all layouts.

    Stability:
        caller
    """

    name: str
    """
    The name of the attribute.
    """
    _spec: ColumnSpec
    _table: pa.Table
    _vocab: Vocabulary
    _selected: pa.Int32Array | None = None
    _cached_array: pa.Array | pa.ChunkedArray | None = None

    def __init__(
        self,
        name: str,
        spec: ColumnSpec,
        table: pa.Table,
        vocab: Vocabulary,
        rows: pa.Int32Array | None,
    ):
        self.name = name
        self._spec = spec
        self._table = table
        self._vocab = vocab
        self._selected = rows

    def ids(self) -> IDArray:
        """
        Get the entity IDs for the rows in this attribute's values.
        """
        if self._selected is None:
            return self._vocab.ids()
        else:
            return self._vocab.ids(self._selected.to_numpy())

    def id_index(self) -> pd.Index:
        """
        Get the entity IDs as a Pandas index.
        """
        if self._selected is None:
            return self._vocab.index
        else:
            return self._vocab.index[self._selected.to_numpy()]

    def numbers(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """
        Get the entity numbers for the rows in this attribute's values.
        """
        if self._selected is None:
            return np.arange(self._table.num_rows, dtype=np.int32)
        else:
            return self._selected.to_numpy()

    @property
    def dim_names(self) -> list[str] | None:
        """
        Get the names attached to this attribute's dimensions.

        .. note::

            Only applicable to vector and sparse attributes.
        """
        return None

    @property
    def is_scalar(self) -> bool:
        """
        Query whether this attribute is scalar.
        """
        return self._spec.layout == AttrLayout.SCALAR

    @property
    def is_list(self) -> bool:
        """
        Query whether this attribute is a list.
        """
        return self._spec.layout == AttrLayout.LIST

    @property
    def is_vector(self) -> bool:
        """
        Query whether this attribute is a dense vector.
        """
        return self._spec.layout == AttrLayout.VECTOR

    @property
    def is_sparse(self) -> bool:
        """
        Query whether this attribute is a sparse vector.
        """
        return self._spec.layout == AttrLayout.SPARSE

    def pandas(
        self, *, missing: Literal["null", "omit"] = "null"
    ) -> pd.Series | pd.DataFrame:  # pragma: nocover
        raise NotImplementedError()

    def numpy(self) -> NDArray[Any]:
        """
        Get the attribute values as a NumPy array.

        .. note::
            Undefined attribute values may have undefined contents; they will
            _usually_ be ``NaN`` or similar, but this is not fully guaranteed.
        """
        return self.arrow().to_numpy()

    def arrow(self) -> pa.Array[Any] | pa.ChunkedArray[Any]:
        """
        Get the attribute values as an Arrow array.
        """
        if self._cached_array is None:
            col = self._table.column(self.name)
            if self._selected is not None:
                col = col.take(self._selected)
            self._cached_array = col

        return self._cached_array

    def scipy(self) -> NDArray[Any] | csr_array:
        """
        Get this attribute as a SciPy sparse array (if it is sparse), or a NumPy
        array if it is dense.
        """
        return self.numpy()

    def torch(self) -> torch.Tensor:
        return safe_tensor(self.numpy())

    def drop_null(self):
        """
        Subset this attribute set to only entities for which it is defined.
        """
        col = self._table.column(self.name)
        valid = col.is_valid().combine_chunks()
        if self._selected is not None:
            valid = valid.take(self._selected)
            selected = self._selected.filter(valid)
        else:
            selected = np.arange(len(self._vocab), dtype=np.int32)
            selected = pa.array(selected)
            assert isinstance(selected, pa.Int32Array)
            selected = selected.filter(valid)

        return self.__class__(self.name, self._spec, self._table, self._vocab, selected)

    def __len__(self):
        if self._selected is not None:
            return len(self._selected)
        else:
            return len(self._vocab)


class ScalarAttributeSet(AttributeSet):
    def pandas(self, *, missing: Literal["null", "omit"] = "null") -> pd.Series[Any]:
        arr = self.arrow()
        mask = arr.is_valid()
        if missing == "null" and pc.all(mask).as_py():
            return pd.Series(arr.to_numpy(zero_copy_only=False), index=self.ids())
        else:
            mask = mask.to_numpy(zero_copy_only=False)
            return pd.Series(arr.drop_null().to_numpy(zero_copy_only=False), index=self.ids()[mask])


class ListAttributeSet(AttributeSet):
    def pandas(self, *, missing: Literal["null", "omit"] = "null") -> pd.Series[Any]:
        arr = self.arrow()
        mask = arr.is_valid()
        if missing == "null":
            s = arr.to_pandas()
            s.index = pd.Index(self.ids())
            return s
        elif missing == "omit":
            mask = mask.to_numpy(zero_copy_only=False)
            return pd.Series(arr.drop_null().to_numpy(zero_copy_only=False), index=self.ids()[mask])


class VectorAttributeSet(AttributeSet):
    _names: list[str] | None = None
    _size: int | None = None

    @property
    def dim_names(self) -> list[str] | None:
        """
        Get the names of the dimensions of this vector.  I.e., if the vector
        is storing weights attached to tags, the dimension names will be the
        tags.
        """
        if self._names is None:
            field = self._table.field(self.name)
            meta = field.metadata
            if meta is None:
                return None

            nstr = meta.get(b"lenskit:names", None) if meta else None
            if nstr is not None:
                self._names = json.loads(nstr)

        return self._names

    @property
    def vector_size(self) -> int:
        """
        Get the size (dimensionality) of this vector attribute.
        """
        assert self._spec.vector_size is not None, "vector column has no size"
        return self._spec.vector_size

    def arrow(self) -> pa.Array[Any] | pa.ChunkedArray[Any]:
        """
        Get the attribute values as an Arrow array.
        """

        col = self._table.column(self.name)
        if self._selected is not None:
            col = col.take(self._selected)

        if pa.types.is_fixed_size_list(col.type):
            return col
        elif col.null_count == 0:
            if isinstance(col, pa.ChunkedArray):
                return pa.chunked_array(
                    [
                        pa.FixedSizeListArray.from_arrays(c.values, self.vector_size)
                        for c in col.chunks
                    ]
                )
            elif isinstance(col, pa.ListArray):
                return pa.FixedSizeListArray.from_arrays(col.values, self.vector_size)
            else:  # pragma: nocover
                raise TypeError("unexpected array type")
        else:
            # now things get tricky.
            if isinstance(col, pa.ChunkedArray):
                col = col.combine_chunks()
            assert isinstance(col, pa.ListArray)
            fixed = pa.FixedSizeListArray.from_arrays(col.values, self.vector_size)
            mat = pa.nulls(len(col), type=fixed.type)
            valid = col.is_valid()
            assert pc.all(pc.equal(col.value_lengths().filter(valid), self.vector_size)).as_py()
            col = _replace_vectors(mat, valid, fixed)

        return col

    def numpy(self) -> np.ndarray[tuple[int, int], Any]:
        arr = self.arrow()
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        assert isinstance(arr, pa.FixedSizeListArray)
        mat = arr.values.to_numpy(zero_copy_only=False).reshape((len(arr), arr.type.list_size))
        return mat

    def pandas(self, *, missing: Literal["null", "omit"] = "null") -> pd.DataFrame:
        arr = self.arrow()
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        assert isinstance(arr, pa.FixedSizeListArray)

        ids = self.ids()

        mask = arr.is_valid()
        if missing == "omit" and not pc.all(mask).as_py():
            arr = arr.drop_null()
            ids = ids[mask.to_numpy(zero_copy_only=False)]

        mat = arr.values.to_numpy().reshape((len(arr), arr.type.list_size))
        return pd.DataFrame(mat, index=ids, columns=self.dim_names)


class SparseAttributeSet(AttributeSet):
    _names: list[str] | None = None

    @property
    def dim_names(self) -> list[str] | None:
        """
        Get the names of the dimensions of this vector.  I.e., if the vector
        is storing weights attached to tags, the dimension names will be the
        tags.
        """

        if self._names is None:
            field = self._table.field(self.name)
            meta = field.metadata
            if meta is None:
                return None

            nstr = meta.get(b"lenskit:names", None) if meta else None
            if nstr is not None:
                self._names = json.loads(nstr)

        return self._names

    def arrow(self) -> SparseRowArray:
        arr = super().arrow()
        if not isinstance(arr, SparseRowArray):
            dim = self._spec.vector_size
            if isinstance(arr, pa.ChunkedArray):
                arr = arr.combine_chunks()
            arr = SparseRowArray.from_array(arr, dim)
            self._cached_array = arr

        assert isinstance(self._cached_array, SparseRowArray)
        return self._cached_array

    def numpy(self) -> np.ndarray[tuple[int, int], Any]:
        raise NotImplementedError("sparse attributes cannot be retrieved as numpy")

    def scipy(self) -> csr_array:
        col = self.arrow()

        return col.to_scipy()

    def torch(self) -> torch.Tensor:
        col = self.arrow()

        return col.to_torch()


def _replace_vectors(
    arr: pa.FixedSizeListArray, mask: pa.BooleanArray, values: pa.FixedSizeListArray
):
    size = arr.type.list_size
    assert values.type.list_size == size
    assert values.null_count == 0
    in_valid = arr.is_valid()
    out_valid = pc.or_(in_valid, mask)

    value_mask = np.repeat(np.asarray(mask), size)
    value_mask = pa.array(value_mask)
    new_vals = pc.replace_with_mask(arr.values, value_mask, values.values)
    return pa.FixedSizeListArray.from_arrays(new_vals, size, mask=pc.invert(out_valid))
