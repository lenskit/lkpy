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
    _spec: ColumnSpec
    _table: pa.Table
    _vocab: Vocabulary
    _selected: pa.Int32Array | None = None

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

    def numbers(self) -> np.ndarray[int, np.dtype[np.int32]]:
        """
        Get the entity numbers for the rows in this attribute's values.
        """
        if self._selected is None:
            return np.arange(self._table.num_rows, dtype=np.int32)
        else:
            return self._selected.to_numpy()

    @property
    def names(self) -> list[str] | None:
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
        return self.arrow().to_numpy()

    def arrow(self) -> pa.Array[Any] | pa.ChunkedArray[Any]:
        col = self._table.column(self.name)
        if self._selected is not None:
            col = col.take(self._selected)

        return col

    def scipy(self) -> NDArray[Any] | csr_array:
        """
        Get this attribute as a SciPy sparse array (if it is sparse), or a NumPy
        array if it is dense.
        """
        return self.numpy()

    def torch(self) -> torch.Tensor:
        return torch.from_numpy(self.numpy())


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
        if missing == "null" and pc.all(mask).as_py():
            return pd.Series(arr.to_numpy(zero_copy_only=False), index=self.ids())
        else:
            mask = mask.to_numpy(zero_copy_only=False)
            return pd.Series(arr.drop_null().to_numpy(zero_copy_only=False), index=self.ids()[mask])


class VectorAttributeSet(AttributeSet):
    _names: list[str] | None = None

    @property
    def names(self) -> list[str] | None:
        if self._names is None:
            field = self._table.field(self.name)
            meta = field.metadata
            nstr = meta.get(b"lenskit:names", None) if meta else None
            if nstr is not None:
                self._names = json.loads(nstr)

        return self._names

    def numpy(self) -> np.ndarray[tuple[int, int], Any]:
        arr = self.arrow()
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        assert isinstance(arr, pa.FixedSizeListArray)
        mat = arr.values.to_numpy().reshape((len(arr), arr.type.list_size))
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
        return pd.DataFrame(mat, index=ids, columns=self.names)


class SparseAttributeSet(AttributeSet):
    _names: list[str] | None = None

    @property
    def names(self) -> list[str] | None:
        if self._names is None:
            field = self._table.field(self.name)
            meta = field.metadata
            nstr = meta.get(b"lenskit:names", None) if meta else None
            if nstr is not None:
                self._names = json.loads(nstr)

        return self._names

    def numpy(self) -> np.ndarray[tuple[int, int], Any]:
        raise NotImplementedError("sparse attributes cannot be retrieved as numpy")

    def scipy(self) -> csr_array:
        col = self.arrow()
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        assert isinstance(col, pa.ListArray)

        rowptr = col.offsets.to_numpy()
        entries = col.values
        assert isinstance(entries, pa.StructArray)
        indices = entries.field("index").to_numpy(zero_copy_only=False)
        values = entries.field("value").to_numpy(zero_copy_only=False)

        field = self._table.field(self.name)
        meta = field.metadata
        assert meta is not None
        ncol = meta[b"lenskit:ncol"]
        ncol = int(ncol)

        return csr_array((values, indices, rowptr), shape=(len(col), ncol))

    def torch(self) -> torch.Tensor:
        csr = self.scipy()
        return torch.sparse_csr_tensor(
            crow_indices=np.require(csr.indptr, requirements="W"),
            col_indices=np.require(csr.indices, requirements="W"),
            values=np.require(csr.data, requirements="W"),
            size=csr.shape,
        )
