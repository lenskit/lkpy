"""
Data attribute accessors.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import torch
from numpy.typing import NDArray
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
        case _:
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
        return pd.DataFrame(mat, index=ids)
