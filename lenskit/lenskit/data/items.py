# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Primary item-list abstraction.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from numpy.typing import ArrayLike, NDArray
from typing_extensions import (
    Any,
    Literal,
    LiteralString,
    Sequence,
    TypeAlias,
    cast,
)

from .checks import check_1d
from .mtarray import MTArray, MTGenericArray
from .types import IDArray, IDSequence
from .vocab import Vocabulary

Backend: TypeAlias = Literal["numpy", "torch"]


class ItemList:
    """
    Representation of a (usually ordered) list of items, possibly with scores
    and other associated data; many components take and return item lists.  Item
    lists are to be treated as **immutable** — create a new list with modified
    data, do not do in-place modifications of the list itself or the arrays or
    data frame it returns.

    An item list logically a list of rows, each of which is an item with
    multiple fields.  A designated field, ``score``, is available through the
    :meth:`scores` method, and is always single-precision floating-point.

    Item lists can be subset as an array (e.g. ``items[selector]``), where
    integer indices (or arrays thereof), boolean arrays, and slices are allowed
    as selectors.

    When an item list is pickled, it is pickled compactly but only for CPUs: the
    vocabulary is dropped (after ensuring both IDs and numbers are computed),
    and all arrays are pickled as NumPy arrays.  This makes item lists compact
    to serialize and transmit, but does mean that that serializing an item list
    whose scores are still on the GPU will deserialize on the CPU in the
    receiving process.  This is usually not a problem, because item lists are
    typically used for small lists of items, not large data structures that need
    to remain in shared memory.

    .. note::

        Naming for fields and accessor methods is tricky, because the usual
        convention for a data frame is to use singular column names (e.g.
        “item_id”, “score”) instead of plural (“item_ids”, “scores”) — the data
        frame, like a database table, is a list of instances, and the column
        names are best interpreted as naming attributes of individual instances.

        However, when working with a list of e.g. item IDs, it is more natural —
        at least to this author — to use plural names: ``item_ids``.  Since this
        class is doing somewhat double-duty, representing a list of items along
        with associated data, as well as a data frame of columns representing
        items, the appropriate naming is not entirely clear.  The naming
        convention in this class is therefore as follows:

        * Field names are singular (``item_id``, ``score``).
        * Named accessor methods are plural (:meth:`item_ids`, :meth:`scores`).
        * Both singular and plural forms are accepted for item IDs numbers, and
          scores in the keyword arguments.  Other field names should be
          singular.

    .. todo::

        Right now, selection / subsetting only happens on the CPU, and will move
        data to the CPU for the subsetting operation.  There is no reason, in
        principle, why we cannot subset on GPU.  Future revisions may add
        support for this.

    Args:
        source:
            A source item list. If provided and an :class:`ItemList`, its fields
            and data are used to initialize any aspects of the item list that
            are not provided in the other arguments.  Otherwise, it is
            interpreted as ``item_ids``.
        item_ids:
            A list or array of item identifiers. ``item_id`` is accepted as an
            alternate name.
        item_nums:
            A list or array of item numbers. ``item_num`` is accepted as an
            alternate name.
        vocabulary:
            A vocabulary to translate between item IDs and numbers.
        ordered:
            Whether the list has a meaningful order.
        scores:
            An array of scores for the items.  Pass the value ``False`` to
            remove the scores when copying from a source list.
        fields:
            Additional fields, such as ``score`` or ``rating``.  Field names
            should generally be singular; the named keyword arguments and
            accessor methods are plural for readability (“get the list of item
            IDs”).  Pass the value ``False`` to remove the field when copying
            from a source list.
    """

    ordered: bool = False
    "Whether this list has a meaningful order."
    _len: int
    _ids: IDArray | None = None
    _numbers: MTArray[np.int32] | None = None
    _vocab: Vocabulary | None = None
    _ranks: MTArray[np.int32] | None = None
    _fields: dict[str, MTGenericArray]

    def __init__(
        self,
        source: ItemList | IDSequence | None = None,
        *,
        item_ids: IDSequence | None = None,
        item_nums: NDArray[np.int32] | pd.Series[int] | Sequence[int] | ArrayLike | None = None,
        vocabulary: Vocabulary | None = None,
        ordered: bool | None = None,
        scores: NDArray[np.generic]
        | torch.Tensor
        | ArrayLike
        | Literal[False]
        | np.floating
        | float
        | None = None,
        **fields: NDArray[np.generic] | torch.Tensor | ArrayLike | Literal[False],
    ):
        if isinstance(source, ItemList):
            self.__dict__.update(source.__dict__)
            eff_fields = source._fields | fields
        else:
            eff_fields = fields

        if ordered is not None:
            self.ordered = ordered
        elif source is None:
            self.ordered = False

        if vocabulary is not None:
            if not isinstance(vocabulary, Vocabulary):  # pragma: nocover
                raise TypeError(f"expected Vocabulary, got {type(vocabulary)}")
            self._vocab = vocabulary

        # handle aliases for item ID/number columns
        if item_ids is None and "item_id" in fields:
            item_ids = np.asarray(cast(Any, fields["item_id"]))
        if source is not None and not isinstance(source, ItemList):
            if item_ids is None:
                item_ids = source
                source = None
            else:
                raise ValueError("cannot specify both item_ids & item ID source")

        if item_nums is None and "item_num" in fields:
            item_nums = np.asarray(cast(Any, fields["item_num"]))
            if not issubclass(item_nums.dtype.type, np.integer):
                raise TypeError("item numbers not integers")

        # empty list
        if item_ids is None and item_nums is None and source is None:
            self._ids = np.ndarray(0, dtype=np.int32)
            self._numbers = MTArray(np.ndarray(0, dtype=np.int32))
            self._len = 0

        if item_ids is not None:
            if len(item_ids):
                self._ids = np.asarray(item_ids)
            else:
                self._ids = np.ndarray(0, dtype=np.int32)
            if not issubclass(self._ids.dtype.type, (np.integer, np.str_, np.bytes_, np.object_)):
                raise TypeError(f"item IDs not integers or bytes (type: {self._ids.dtype})")

            check_1d(self._ids, label="item_ids")
            self._len = len(item_ids)
            # clear numbers if we got them from the source
            if source is not None and source._numbers is not None:
                del self._numbers

        if item_nums is not None:
            if not len(item_nums):  # type: ignore
                item_nums = np.ndarray(0, dtype=np.int32)
            self._numbers = MTArray(item_nums)
            check_1d(self._numbers, getattr(self, "_len", None), label="item_nums")
            self._len = self._numbers.shape[0]
            # clear IDs if we got them from the source
            if source is not None and source._ids is not None:
                del self._ids

        if scores is False:  # check 'is False' to distinguish from None
            scores = None
        else:
            if scores is None and "score" in eff_fields:
                scores = np.require(eff_fields["score"], dtype=np.float32)
            elif scores is not None:
                if "score" in fields:  # pragma: nocover
                    raise ValueError("cannot specify both scores= and score=")

                if np.isscalar(scores):
                    scores = np.full(self._len, scores, dtype=np.float32)
                else:
                    scores = np.require(scores, np.float32)

        # convert fields and drop singular ID/number aliases
        self._fields = {}
        if scores is not None:
            self._fields["score"] = check_1d(MTArray(scores), self._len, label="score")

        for name, data in eff_fields.items():
            if name not in ("item_id", "item_num", "score") and data is not False:
                self._fields[name] = check_1d(MTArray(data), self._len, label=name)

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, *, vocabulary: Vocabulary | None = None, keep_user: bool = False
    ) -> ItemList:
        """
        Create a item list from a Pandas data frame.  The frame should have
        ``item_num`` and/or ``item_id`` columns to identify the items; other
        columns (e.g. ``score`` or ``rating``) are added as fields. If the data
        frame has user columns (``user_id`` or ``user_num``), those are dropped
        by default.

        Args:
            df:
                The data frame to turn into an item list.
            vocabulary:
                The item vocabulary.
            keep_user:
                If ``True``, keeps user ID/number columns instead of dropping them.
        """
        ids = df["item_id"].values if "item_id" in df.columns else None
        nums = df["item_num"].values if "item_num" in df.columns else None
        if ids is None and nums is None:
            raise TypeError("data frame must have at least one of item_id, item_num columns")

        to_drop = ["item_id", "item_num"]
        if not keep_user:
            to_drop += ["user_id", "user_num"]

        if "rank" in df.columns:
            ranks = df["rank"]
            to_drop += "rank"
            if ranks.iloc[0] != 1:
                raise ValueError("item ranks do not start with 1")
            if np.any(np.diff(ranks) != 1):
                raise ValueError("item ranks not consecutive")
        else:
            ranks = None

        fields = {f: df[f].values for f in df.columns if f not in to_drop}
        items = cls(
            item_ids=ids,  # type: ignore
            item_nums=nums,  # type: ignore
            vocabulary=vocabulary,
            ordered=ranks is not None,
            **fields,  # type: ignore
        )
        if ranks is not None:
            items._ranks = MTArray(np.require(ranks.values, np.int32))  # type: ignore
        return items

    @classmethod
    def from_arrow(
        cls,
        tbl: pa.StructArray | pa.Table,
        *,
        vocabulary: Vocabulary | None = None,
    ) -> ItemList:
        """
        Create a item list from a Pandas table or structured array.  The table
        should have ``item_num`` and/or ``item_id`` columns to identify the
        items; other columns (e.g. ``score`` or ``rating``) are added as fields.
        If the data frame has user columns (``user_id`` or ``user_num``), those
        are dropped by default.

        Args:
            tbl:
                The Arrow table or array to convert to an item list.
            vocabulary:
                The item vocabulary.
        """
        if isinstance(tbl, pa.Table):
            tbl = tbl.to_struct_array()  # type: ignore

        if isinstance(tbl, pa.ChunkedArray):
            tbl = tbl.combine_chunks()  # type: ignore
        assert isinstance(tbl, pa.StructArray)
        assert isinstance(tbl.type, pa.StructType)

        names = tbl.type.names

        ids = None
        nums = None
        if "item_id" in names:
            ids = tbl.field("item_id")
        if "item_num" in names:
            nums = tbl.field("item_num")

        if ids is None and nums is None:
            raise TypeError("data table must have at least one of item_id, item_num columns")

        to_drop = ["item_id", "item_num"]

        if "rank" in names:
            ranks = tbl.field("rank").to_numpy()
            to_drop.append("rank")
            if ranks[0] != 1:
                raise ValueError("item ranks do not start with 1")
            if np.any(np.diff(ranks) != 1):
                raise ValueError("item ranks not consecutive")
        else:
            ranks = None

        fields = {c: tbl.field(c).to_numpy() for c in names if c not in to_drop}
        items = cls(
            item_ids=ids,  # type: ignore
            item_nums=nums,  # type: ignore
            vocabulary=vocabulary,
            ordered=ranks is not None,
            **fields,  # type: ignore
        )
        if ranks is not None:
            items._ranks = MTArray(np.require(ranks, np.int32))  # type: ignore
        assert len(items) == len(tbl)
        return items

    @classmethod
    def from_vocabulary(cls, vocab: Vocabulary) -> ItemList:
        return ItemList(item_ids=vocab.ids(), item_nums=np.arange(len(vocab)), vocabulary=vocab)

    def clone(self) -> ItemList:
        """
        Make a shallow copy of the item list.
        """
        return ItemList(
            item_ids=self._ids,
            item_nums=self._numbers,
            vocabulary=self._vocab,
            ordered=self.ordered,
            **self._fields,  # type: ignore
        )

    @property
    def vocabulary(self) -> Vocabulary | None:
        "Get the item list's vocabulary, if available."
        return self._vocab

    def ids(self) -> IDArray:
        """
        Get the item IDs.

        Returns:
            An array of item identifiers.

        Raises:
            RuntimeError: if the item list was not created with IDs or a :class:`Vocabulary`.
        """
        if self._ids is None:
            if self._vocab is None:
                raise RuntimeError("item IDs not available (no IDs or vocabulary provided)")
            assert self._numbers is not None
            self._ids = self._vocab.ids(self._numbers.numpy())

        return self._ids

    @overload
    def numbers(
        self,
        format: Literal["numpy"] = "numpy",
        *,
        vocabulary: Vocabulary | None = None,
        missing: Literal["error", "negative"] = "error",
    ) -> NDArray[np.int32]: ...
    @overload
    def numbers(
        self,
        format: Literal["torch"],
        *,
        vocabulary: Vocabulary | None = None,
        missing: Literal["error", "negative"] = "error",
    ) -> torch.Tensor: ...
    @overload
    def numbers(
        self,
        format: LiteralString = "numpy",
        *,
        vocabulary: Vocabulary | None = None,
        missing: Literal["error", "negative"] = "error",
    ) -> ArrayLike: ...
    def numbers(
        self,
        format: LiteralString = "numpy",
        *,
        vocabulary: Vocabulary | None = None,
        missing: Literal["error", "negative"] = "error",
    ) -> ArrayLike:
        """
        Get the item numbers.

        Args:
            format:
                The array format to use.
            vocabulary:
                A alternate vocabulary for mapping IDs to numbers.  If provided,
                then the item list must have IDs (either stored, or through a
                vocabulary).

        Returns:
            An array of item numbers.

        Raises:
            RuntimeError: if the item list was not created with numbers or a
            :class:`Vocabulary`.
        """
        if vocabulary is not None and not vocabulary.compatible_with_numbers_from(self._vocab):
            # we need to translate vocabulary
            ids = self.ids()
            return vocabulary.numbers(ids, missing=missing)

        if self._numbers is None:
            if self._vocab is None:
                raise RuntimeError("item numbers not available (no IDs or vocabulary provided)")
            assert self._ids is not None
            self._numbers = MTArray(self._vocab.numbers(self._ids, missing="negative"))

        if missing == "error" and np.any(self._numbers.to("numpy") < 0):
            raise KeyError("item IDs")
        return self._numbers.to(format)

    @overload
    def scores(self, format: Literal["numpy"] = "numpy") -> NDArray[np.floating] | None: ...
    @overload
    def scores(self, format: Literal["torch"]) -> torch.Tensor | None: ...
    @overload
    def scores(
        self, format: Literal["pandas"], *, index: Literal["ids", "numbers"] | None = None
    ) -> pd.Series | None: ...
    @overload
    def scores(self, format: LiteralString = "numpy") -> ArrayLike | None: ...
    def scores(self, format: LiteralString = "numpy", **kwargs) -> ArrayLike | None:
        """
        Get the item scores (if available).
        """
        return self.field("score", format, **kwargs)

    @overload
    def ranks(self, format: Literal["numpy"] = "numpy") -> NDArray[np.int32] | None: ...
    @overload
    def ranks(self, format: Literal["torch"]) -> torch.Tensor | None: ...
    @overload
    def ranks(self, format: LiteralString = "numpy") -> ArrayLike | None: ...
    def ranks(self, format: LiteralString = "numpy") -> ArrayLike | None:
        """
        Get an array of ranks for the items in this list, if it is ordered.
        Unordered lists have no ranks.  The ranks are based on the order in the
        list, **not** on the score.

        Item ranks start with **1**, for compatibility with common practice in
        mathematically defining information retrieval metrics and operations.

        Returns:
            An array of item ranks, or ``None`` if the list is unordered.
        """
        if not self.ordered:
            return None

        if self._ranks is None:
            self._ranks = MTArray(np.arange(1, self._len + 1, dtype=np.int32))

        return self._ranks.to(format)

    @overload
    def field(
        self, name: str, format: Literal["numpy"] = "numpy"
    ) -> NDArray[np.floating] | None: ...
    @overload
    def field(self, name: str, format: Literal["torch"]) -> torch.Tensor | None: ...
    @overload
    def field(
        self,
        name: str,
        format: Literal["pandas"],
        *,
        index: Literal["ids", "numbers"] | None = None,
    ) -> pd.Series | None: ...
    @overload
    def field(self, name: str, format: LiteralString) -> ArrayLike | None: ...
    def field(
        self, name: str, format: LiteralString = "numpy", *, index: LiteralString | None = None
    ) -> ArrayLike | None:
        val = self._fields.get(name, None)
        if val is None:
            return None
        elif format == "pandas":
            idx = None
            vs = val.to("numpy")
            if index == "ids":
                idx = pd.Index(self.ids(), name="item_id")
            elif index == "numbers":
                idx = pd.Index(self.numbers(), name="item_num")
            elif index:  # pragma: nocover
                raise ValueError(f"unsupported Pandas index {index}")
            return pd.Series(vs, index=idx)
        else:
            return val.to(format)

    def to_df(self, *, ids: bool = True, numbers: bool = True) -> pd.DataFrame:
        """
        Convert this item list to a Pandas data frame.  It has the following columns:

        * ``item_id`` — the item IDs (if available and ``ids=True``)
        * ``item_num`` — the item numbers (if available and ``numbers=True``)
        * ``score`` — the item scores
        * ``rank`` — the item ranks (if the list is ordered)
        * all other defined fields, using their field names
        """
        cols = {}
        if ids and (self._ids is not None or self._vocab is not None):
            cols["item_id"] = self.ids()
        if numbers and (self._numbers is not None or self._vocab is not None):
            cols["item_num"] = self.numbers()
        # we need to have numbers or ids, or it makes no sense
        if "item_id" not in cols and "item_num" not in cols:
            if ids and not numbers:
                raise RuntimeError("item list has no vocabulary, cannot compute IDs")
            elif numbers and not ids:
                raise RuntimeError("item list has no vocabulary, cannot compute numbers")
            else:
                raise RuntimeError("cannot create item data frame without identifiers or numbers")

        if "score" in self._fields:
            cols["score"] = self.scores()
        if self.ordered:
            cols["rank"] = self.ranks()
        # add remaining fields
        cols.update((k, v.numpy()) for (k, v) in self._fields.items() if k != "score")
        return pd.DataFrame(cols)

    @overload
    def to_arrow(
        self,
        *,
        ids: bool = True,
        numbers: bool = False,
        type: Literal["table"] = "table",
        columns: dict[str, pa.DataType] | None = None,
    ) -> pa.Table: ...
    @overload
    def to_arrow(
        self,
        *,
        ids: bool = True,
        numbers: bool = False,
        type: Literal["array"],
        columns: dict[str, pa.DataType] | None = None,
    ) -> pa.StructArray: ...
    def to_arrow(
        self,
        *,
        ids: bool = True,
        numbers: bool = False,
        type: Literal["table", "array"] = "table",
        columns: dict[str, pa.DataType] | None = None,
    ):
        """
        Convert the item list to a Pandas table.
        """
        arrays = []
        names = []
        if columns is not None and len(self) == 0:
            arrays = [pa.array([], ft) for ft in columns.values()]
            names = list(columns.keys())
        else:
            if (ids or (columns is not None and "item_id" in columns)) and (
                self._ids is not None or self._vocab is not None
            ):
                arrays.append(self.ids())
                names.append("item_id")
            if (numbers or (columns is not None and "item_num" in columns)) and (
                self._numbers is not None or self._vocab is not None
            ):
                arrays.append(self.numbers())
                names.append("item_num")

            # we need to have numbers or ids, or it makes no sense
            if "item_id" not in names and "item_num" not in names:
                if ids and not numbers:
                    raise RuntimeError("item list has no vocabulary, cannot compute IDs")
                elif numbers and not ids:
                    raise RuntimeError("item list has no vocabulary, cannot compute numbers")
                else:
                    raise RuntimeError(
                        "cannot create item data frame without identifiers or numbers"
                    )

            if columns is not None:
                for fn, ft in columns.items():
                    if fn in ("item_id", "item_num"):
                        continue

                    names.append(fn)
                    col = self._fields.get(fn, None)
                    if col is not None:
                        arrays.append(col.numpy())
                    else:
                        arrays.append(pa.nulls(len(self), ft))
            else:
                if "score" in self._fields:
                    arrays.append(self.scores())
                    names.append("score")
                if self.ordered:
                    arrays.append(self.ranks())
                    names.append("rank")
                # add remaining fields
                for k, v in self._fields.items():
                    if k == "score" or k == "rank":
                        continue

                    arrays.append(v.numpy())
                    names.append(k)

        if type == "table":
            return pa.Table.from_arrays(arrays, names)
        elif type == "array":
            return pa.RecordBatch.from_arrays(arrays, names).to_struct_array()
        else:  # pragma: nocover
            raise ValueError(f"unsupported target type {type}")

    def arrow_types(self, *, ids: bool = True, numbers: bool = False) -> dict[str, pa.DataType]:
        """
        Get the Arrow data types for this item list.
        """
        types: dict[str, pa.DataType] = {}
        if len(self) == 0:
            return types

        if ids:
            if self._ids is not None:
                types["item_id"] = pa.from_numpy_dtype(self._ids.dtype)
            elif self._vocab is not None:
                types["item_id"] = pa.from_numpy_dtype(self._vocab.ids().dtype)

        if numbers and (self._numbers is not None or self._vocab is not None):
            types["item_num"] = pa.int32()

        for name, f in self._fields.items():
            types[name] = pa.from_numpy_dtype(f.numpy().dtype)

        return types

    def __len__(self):
        return self._len

    def __getitem__(
        self,
        sel: NDArray[np.bool_] | NDArray[np.integer] | Sequence[int] | torch.Tensor | int | slice,
    ) -> ItemList:
        """
        Subset the item list.

        .. todo::
            Support on-device masking.

        Args:
            sel:
                The items to select. Can be either a Boolean array of the same
                length as the list that is ``True`` to indicate selected items,
                or an array of indices of the items to retain (in order in the
                list, starting from 0).
        """
        if np.isscalar(sel):
            sel = np.array([sel])
        elif not isinstance(sel, slice):
            sel = np.asarray(sel)

        # sel is now a selection array, or it is a slice. numpy supports both.
        iids = self._ids[sel] if self._ids is not None else None
        nums = self._numbers.numpy()[sel] if self._numbers is not None else None
        flds = {n: f.numpy()[sel] for (n, f) in self._fields.items()}
        return ItemList(
            item_ids=iids,
            item_nums=nums,
            vocabulary=self._vocab,
            ordered=self.ordered,
            **flds,  # type: ignore
        )

    def __getstate__(self) -> dict[str, object]:
        state: dict[str, object] = {"ordered": self.ordered, "len": self._len}
        if self._ids is not None:
            state["ids"] = self._ids
        elif self._vocab is not None:
            # compute the IDs so we can save them
            state["ids"] = self.ids()

        if self._numbers is not None:
            state["numbers"] = self._numbers.numpy()
        elif self._vocab is not None:
            state["numbers"] = self.numbers()

        state.update(("field_" + k, v.numpy()) for (k, v) in self._fields.items())
        return state

    def __setstate__(self, state: dict[str, Any]):
        self.ordered = state["ordered"]
        self._len = state["len"]
        self._ids = state.get("ids", None)
        if "numbers" in state:
            self._numbers = MTArray(state["numbers"])
        self._fields = {k[6:]: MTArray(v) for (k, v) in state.items() if k.startswith("field_")}

    def __str__(self) -> str:
        return f"<ItemList of {self._len} items>"

    def __repr__(self) -> str:
        fnames = ", ".join(self._fields.keys())
        nf = len(self._fields)
        return f"<ItemList of {self._len} items with {nf} fields ({fnames})>"
