# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Primary item-list abstraction.
"""

from __future__ import annotations

import io
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import torch
from numpy.typing import ArrayLike, NDArray
from typing_extensions import (
    Any,
    Literal,
    LiteralString,
    overload,
)

from lenskit._accel import data as _data_accel
from lenskit.diagnostics import DataWarning
from lenskit.stats import argtopn

from .arrow import get_indexer
from .checks import check_1d
from .mtarray import MTArray, MTGenericArray
from .types import IDArray, IDSequence, NPVector
from .vocab import Vocabulary

ILIndexer = (
    np.ndarray[tuple[int], np.dtype[np.bool_]]
    | NPVector[np.integer]
    | Sequence[int]
    | torch.Tensor
    | pa.BooleanArray
    | pa.Int32Array
    | int
    | slice
)


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

    In a few places, the “items” in an item list may be other entities, such as
    users, tags, authors, etc.  This seems less confusing than calling it
    ``EntityList``, but having a very different use case and feature set than
    :class:`~lenskit.data.EntitySet`.

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

    Stability:
        Caller
    """

    ordered: bool = False
    "Whether this list has a meaningful order."

    _len: int
    "Length of the item list."

    # storage for individual components of the item list
    _ids: pa.Array | None = None
    _ids_numpy: IDArray | None = None
    _mask_cache: tuple[ItemList, np.ndarray[tuple[int], np.dtype[np.bool_]]] | None = None
    _numbers: MTArray[np.int32] | None = None
    _vocab: Vocabulary | None = None
    _ranks: MTArray[np.int32] | None = None
    _fields: dict[str, MTGenericArray]

    @overload
    def __init__(
        self,
        source: ItemList,
        *,
        ordered: bool | None = None,
        vocabulary: Vocabulary | None = None,
        scores: NDArray[np.generic]
        | torch.Tensor
        | ArrayLike
        | Literal[False]
        | np.floating
        | float
        | None = None,
        **fields: NDArray[np.generic] | torch.Tensor | ArrayLike | Literal[False],
    ): ...
    @overload
    def __init__(
        self,
        source: IDSequence | None = None,
        *,
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
    ): ...
    @overload
    def __init__(
        self,
        source: None = None,
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
    ): ...
    def __init__(  # type: ignore
        self,
        source: ItemList | IDSequence | None = None,
        *,
        ordered: bool | None = None,
        vocabulary: Vocabulary | None = None,
        _init_dict: dict[str, Any] | None = None,
        _init_array: pa.StructArray | None = None,
        **kwargs: Any,
    ):
        if _init_dict is not None:
            self.__dict__.update(_init_dict)
            return
        elif _init_array is not None:
            self._init_from_arrow(_init_array, vocabulary, ordered)
        elif isinstance(source, ItemList):
            self.__dict__.update(source.__dict__)
            if ordered is not None:
                self.ordered = ordered
            elif "rank" in kwargs:
                self.ordered = True

            if vocabulary is not None:
                raise ValueError("cannot change vocabulary from item list")

            self._init_fields(**kwargs)
        else:
            self.ordered = ordered if ordered is not None else ("rank" in kwargs)
            self._vocab = vocabulary
            self._init_ids(source, **kwargs)
            self._init_numbers(**kwargs)
            self._init_check_length()
            self._init_fields(**kwargs)

        if vocabulary is not None:
            if not isinstance(vocabulary, Vocabulary):  # pragma: nocover
                raise TypeError(f"expected Vocabulary, got {type(vocabulary)}")
            self._vocab = vocabulary

    def _init_from_arrow(
        self, array: pa.StructArray, vocabulary: Vocabulary | None, ordered: bool | None
    ):
        self._len = len(array)
        self._vocab = vocabulary

        # get the ID and number fields
        try:
            self._ids = array.field("item_id")
        except KeyError:
            pass

        try:
            numbers = array.field("item_num")
        except KeyError:
            pass
        else:
            self._numbers = MTArray(numbers.cast(pa.int32()))

        # set up ranking and ordering
        try:
            ranks = array.field("rank")
        except KeyError:
            self.ordered = ordered or False
        else:
            if ordered is False:
                raise ValueError("table has ranks but ordered=False")
            self.ordered = True
            self._ranks = MTArray(ranks)

        # the rest of the fields can be lazily loaded from the array
        fields = [array.type.field(i) for i in range(array.type.num_fields)]
        self._fields = {
            f.name: MTArray(array.field(i))
            for (i, f) in enumerate(fields)
            if f.name not in ["item_num", "item_id"]
        }

    def _init_ids(
        self,
        source: IDSequence | None,
        *,
        item_ids=None,
        item_id=None,
        **ignore,
    ):
        """
        Initialize the item list's item IDs and length.  The vocabulary should
        already be set.
        """
        # handle aliases for item ID columns
        if source is not None:
            if item_ids is not None or item_id is not None:
                raise ValueError("cannot specify both item_ids & item ID source")
            item_ids = source
        elif item_id is not None:
            if item_ids is not None:
                raise ValueError("cannot specify both item_ids & item_id")
            item_ids = item_id
        elif item_ids is None:
            # no setup
            return

        # handle the item ID type
        if isinstance(item_ids, pa.Array):
            self._ids = item_ids
        elif len(item_ids) > 0:
            try:
                self._ids = pa.array(item_ids)  # type: ignore
            except pa.ArrowInvalid:
                raise TypeError("invalid item ID type or dimension")
            if isinstance(item_ids, np.ndarray):
                self._ids_numpy = item_ids
        else:
            return

        assert self._ids is not None
        idt = self._ids.type
        if not (
            pa.types.is_integer(idt)
            or pa.types.is_string(idt)
            or pa.types.is_binary(idt)
            or pa.types.is_large_string(idt)
            or pa.types.is_large_binary(idt)
        ):
            raise TypeError(f"item IDs not integers or strings (type: {idt})")

        self._len = len(item_ids)

    def _init_numbers(self, item_nums=None, item_num=None, **ignore):
        """
        Initialize the item numbers.
        """

        if item_num is not None:
            if item_nums is not None:
                raise ValueError("cannot specify both item_nums and item_num")
            item_nums = item_num

        length = getattr(self, "_len", None)

        if item_nums is None:
            return

        if not len(item_nums):  # type: ignore
            item_nums = np.ndarray(0, dtype=np.int32)
        if isinstance(item_nums, np.ndarray):
            nk = item_nums.dtype.kind
            if nk != "i" and nk != "u":
                raise TypeError(f"invalid number dtype {item_nums.dtype}")
            item_nums = np.require(item_nums, dtype=np.int32)
        elif isinstance(item_nums, pa.Array):
            if not pa.types.is_integer(item_nums.type):
                raise TypeError(f"invalid number type {item_nums.type}")
            item_nums = item_nums.cast(pa.int32())
        elif torch.is_tensor(item_nums):
            item_nums = item_nums.to(torch.int32)

        self._numbers = MTArray(item_nums)
        check_1d(self._numbers, length, label="item_nums")
        self._len = self._numbers.shape[0]

    def _init_check_length(self):
        """
        Check that we have a correct length, and that lengths match.  Fill in
        empty IDs if we don't have any IDs.

        This is called after setting IDs and numbers, and before setting fields.
        """
        length = getattr(self, "_len", None)
        if self._ids is None and self._numbers is None:
            self._ids = pa.array([], pa.int32())
            self._numbers = MTArray(pa.array([], pa.int32()))
            self._len = 0
        elif length is None:
            if self._ids is not None:
                self._len = len(self._ids)
            elif self._numbers is not None:
                self._len = self._numbers.shape[0]

    def _init_fields(self, *, scores=None, score=None, rank=None, **other):
        if any(k[0] == "_" for k in other.keys()):
            raise ValueError("item list fields cannot start with _")

        fields = getattr(self, "_fields", {}).copy()

        if score is not None:
            if scores is not None:
                raise ValueError("cannot specify both score= and scores=")
            scores = score

        if scores is False and "score" in fields:  # check 'is False' to distinguish from None
            del fields["score"]
        elif scores is not None:
            if np.isscalar(scores):
                scores = np.full(self._len, scores, dtype=np.float32)
            else:
                scores = np.require(scores, np.float32)

            fields["score"] = check_1d(MTArray(scores), self._len, label="scores")

        if rank is not None:
            if not self.ordered:
                warnings.warn(
                    "ranks provided but ordered=False, dropping ranks", DataWarning, stacklevel=3
                )
            else:
                self._ranks = check_1d(
                    MTArray(np.require(rank, np.int32)), self._len, label="ranks"
                )
                fields["rank"] = self._ranks
                if self._len and self._ranks.numpy()[0] != 1:
                    warnings.warn("ranks do not begin with 1", DataWarning, stacklevel=3)

        # convert remaining fields
        for name, fdata in other.items():
            if name in ("item_id", "item_num", "item_ids", "item_nums", "score", "rank"):
                continue
            if fdata is False:
                if name in fields:
                    del fields[name]
                continue

            if fdata is not None:
                fields[name] = check_1d(MTArray(fdata), self._len, label=name)

        self._fields = fields

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

        fields = {f: df[f].values for f in df.columns if f not in to_drop}

        items = cls(
            item_ids=ids,  # type: ignore
            item_nums=nums,  # type: ignore
            vocabulary=vocabulary,
            **fields,  # type: ignore
        )

        return items

    @classmethod
    def from_arrow(
        cls,
        tbl: pa.StructArray | pa.ChunkedArray | pa.Table,
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
            if tbl.num_rows:
                tbl = tbl.to_struct_array()
            else:
                tbl = pa.array([], pa.struct(tbl.schema))

        if isinstance(tbl, pa.ChunkedArray):
            tbl = tbl.combine_chunks()  # type: ignore
        assert isinstance(tbl, pa.StructArray)
        assert isinstance(tbl.type, pa.StructType)

        if hasattr(tbl.type, "names"):
            names = tbl.type.names  # type: ignore
        else:
            names = [tbl.type.field(i).name for i in range(tbl.type.num_fields)]

        if "item_id" not in names and "item_num" not in names:
            raise TypeError("data table must have at least one of item_id, item_num columns")

        return ItemList(_init_array=tbl, vocabulary=vocabulary)

    @classmethod
    def from_vocabulary(cls, vocab: Vocabulary) -> ItemList:
        return ItemList(
            item_ids=vocab.id_array(), item_nums=np.arange(len(vocab)), vocabulary=vocab
        )

    def clone(self) -> ItemList:
        """
        Make a shallow copy of the item list.
        """
        return ItemList(
            _init_dict={
                "_len": self._len,
                "_ids": self._ids,
                "_numbers": self._numbers,
                "_vocab": self._vocab,
                "ordered": self.ordered,
                "_ranks": self._ranks,
                "_fields": self._fields,
            }  # type: ignore
        )  # type: ignore

    @property
    def vocabulary(self) -> Vocabulary | None:
        "Get the item list's vocabulary, if available."
        return self._vocab

    @overload
    def ids(self, *, format: Literal["numpy"] = "numpy") -> IDArray: ...
    @overload
    def ids(self, *, format: Literal["arrow"]) -> pa.Array: ...
    def ids(self, *, format: Literal["numpy", "arrow"] = "numpy"):
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
            self._ids_numpy = self._vocab.ids(self._numbers.numpy())
            self._ids = pa.array(self._ids_numpy)

        match format:
            case "numpy":
                if self._ids_numpy is None:
                    self._ids_numpy = self._ids.to_numpy(zero_copy_only=False)
                return self._ids_numpy
            case "arrow":
                return self._ids
            case _:  # pragma: nocover
                raise ValueError(f"unknown format {format}")

    @overload
    def numbers(
        self,
        format: Literal["numpy"] = "numpy",
        *,
        vocabulary: Vocabulary | None = None,
        missing: Literal["error", "negative"] = "error",
    ) -> NPVector[np.int32]: ...
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
        format: Literal["arrow"],
        *,
        vocabulary: Vocabulary | None = None,
        missing: Literal["error", "negative"] = "error",
    ) -> pa.Array[pa.Int32Scalar]: ...
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
        if vocabulary is not None and vocabulary != self._vocab:
            # we need to translate vocabulary
            ids = self.ids()
            mta = MTArray(vocabulary.numbers(ids, missing=missing))
            return mta.to(format)

        if self._numbers is None:
            if self._vocab is None:
                raise RuntimeError("item numbers not available (no IDs or vocabulary provided)")
            assert self._ids is not None
            self._numbers = MTArray(self._vocab.numbers(self._ids, missing="negative"))

        if missing == "error" and np.any(self._numbers.numpy() < 0):
            raise KeyError("item IDs")
        return self._numbers.to(format)

    @overload
    def scores(self, format: Literal["numpy"] = "numpy") -> NPVector | None: ...
    @overload
    def scores(self, format: Literal["torch"]) -> torch.Tensor | None: ...
    @overload
    def scores(self, format: Literal["arrow"]) -> pa.Array[pa.FloatScalar] | None: ...
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
        return self.field("score", format=format, **kwargs)

    @overload
    def ranks(self, format: Literal["numpy"] = "numpy") -> NDArray[np.int32] | None: ...
    @overload
    def ranks(self, format: Literal["torch"]) -> torch.Tensor | None: ...
    @overload
    def ranks(self, format: Literal["arrow"]) -> pa.Array[pa.Int32Scalar] | None: ...
    @overload
    def ranks(self, format: Literal["pandas"]) -> pd.Series[int] | None: ...
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

        if format == "pandas":
            ranks = self._ranks.to("numpy")
            return pd.Series(ranks, index=self.ids())
        else:
            return self._ranks.to(format)

    @overload
    def field(
        self, name: str, format: Literal["numpy"] = "numpy"
    ) -> NDArray[np.floating] | None: ...
    @overload
    def field(self, name: str, format: Literal["torch"]) -> torch.Tensor | None: ...
    @overload
    def field(self, name: str, format: Literal["arrow"]) -> pa.Array | pa.Tensor | None: ...
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
    ) -> object:
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

    def isin(self, other: ItemList) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """
        Return a boolean mask identifying the items of this list that are in the
        other list.

        This is equivalent to :func:`numpy.isin` applied to the ID arrays, but
        is much more efficient in many cases.
        """
        # cache is-in checks for identical arrays
        if self._mask_cache is not None and self._mask_cache[0] is other:
            return self._mask_cache[1]

        # fast path — try to just use a mask.
        if self.vocabulary is not None:
            # In most common hot-path cases, left is at least as long as right.
            mask = np.zeros(len(self.vocabulary), dtype=np.bool_)
            onums = other.numbers(vocabulary=self.vocabulary, missing="negative")
            onums = onums[onums >= 0]
            mask[onums] = True
            nums = self.numbers()
            result = mask[nums]
        else:
            id_arr = self.ids(format="arrow")
            if pa.types.is_integer(id_arr.type):
                # numpy is quicker than arrow for integer comparisons
                result = np.isin(id_arr.to_numpy(), other.ids())
            else:
                # arrow is quicker than numpy for strings
                try:
                    result = pc.is_in(self.ids(format="arrow"), other.ids(format="arrow")).to_numpy(
                        zero_copy_only=False
                    )
                except pa.ArrowTypeError:
                    # ids are of incompatible types, nothing matches
                    result = np.zeros(len(self), dtype=np.bool_)

        self._mask_cache = (other, result)
        return result

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

        score = self.scores()
        if score is not None:
            cols["score"] = score
        if self.ordered:
            cols["rank"] = self.ranks()

        # add remaining fields
        cols.update((k, v.numpy()) for (k, v) in self._fields.items() if k not in ("score", "rank"))
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
            if columns is None:
                columns = self.arrow_types(ids=ids, numbers=numbers)

            for c_name, c_type in columns.items():
                names.append(c_name)
                if c_name == "item_id":
                    arrays.append(self.ids(format="arrow"))
                elif c_name == "item_num":
                    arrays.append(self.numbers("arrow"))
                elif c_name == "rank":
                    if self.ordered:
                        arrays.append(self.ranks("arrow"))
                    else:
                        warnings.warn("requested rank column for unordered list", DataWarning)
                        arrays.append(pa.nulls(len(self), c_type))
                else:
                    fld = self.field(c_name, format="arrow")

                    if fld is not None:
                        arrays.append(fld)
                    else:
                        warnings.warn(f"unknown field {c_name}", DataWarning)
                        arrays.append(pa.nulls(len(self), c_type))

        if type == "table":
            return pa.Table.from_arrays(arrays, names)
        elif type == "array":
            return pa.StructArray.from_arrays(arrays, names)
        else:  # pragma: nocover
            raise ValueError(f"unsupported target type {type}")

    def arrow_types(self, *, ids: bool = True, numbers: bool = False) -> dict[str, pa.DataType]:
        """
        Get the Arrow data types for this item list.
        """
        types: dict[str, pa.DataType] = {}

        if ids:
            if self._ids is not None:
                types["item_id"] = self._ids.type
            elif self._vocab is not None:
                types["item_id"] = self._vocab.id_array().type

        if numbers and (self._numbers is not None or self._vocab is not None):
            types["item_num"] = pa.int32()

        if self.ordered:
            types["rank"] = pa.int32()

        for n, f in self._fields.items():
            types[n] = f.arrow().type

        return types

    def top_n(self, n: int | None = None, *, scores: str | NPVector | None = None) -> ItemList:
        """
        Get the top _N_ items in this list, sorted in decreasing order.

        If any scores are undefined (``NaN``), those items are excluded.

        Args:
            n:
                The number of items.  If ``None`` or negative, returns all items
                sorted by score.
            scores:
                The name of a field containing the scores, or a NumPy vector of
                scores, for selecting the top _N_.  If ``None``, the item list's
                scores are used.
        Returns:
            An ordered item list containing the top ``n`` items.
        """
        if scores is None:
            scores = self.scores(format="arrow")
            if scores is None:
                raise ValueError("item list has no scores")
        elif isinstance(scores, str):
            scores = self.field(scores, format="arrow")
            if scores is None:
                raise KeyError(scores)
        elif len(scores) != len(self):
            raise ValueError("score array must have same length as items")

        if not isinstance(scores, pa.Array):
            scores = pa.array(scores)

        if n is None or n < 0:
            picked = _data_accel.argsort_descending(scores)
        else:
            picked = argtopn(scores, n)

        return self._take(picked, ordered=True)

    @overload
    def remove(
        self,
        *,
        ids: IDSequence | None = None,
    ) -> ItemList: ...
    @overload
    def remove(
        self,
        *,
        numbers: pa.Int32Array | NPVector[np.integer] | pd.Series[int] | None = None,
    ) -> ItemList: ...
    def remove(
        self,
        *,
        ids: IDSequence | None = None,
        numbers: pa.Int32Array | NPVector[np.integer] | pd.Series[int] | None = None,
    ) -> ItemList:
        """
        Return an item list with the specified items removed.

        The items to remove are not required to be in the list.

        Args:
            ids:
                The item IDs to remove.
            numbers:
                The item numbers to remove.
        """
        if ids is None and numbers is None:
            raise ValueError("must specify one of ids= or numbers=")

        mask = None
        if ids is not None:
            if numbers is not None:
                raise ValueError("must specify only one of ids= or numbers=")

            if self._vocab is not None:
                numbers = self._vocab.numbers(ids, missing="negative")
                numbers = numbers[numbers >= 0]
            else:
                # handle IDs the slow way
                mask = ~self.isin(ItemList(item_ids=ids))

        if mask is None:
            assert numbers is not None

            numbers = MTArray(numbers).numpy()
            mask = np.isin(self.numbers(), numbers, invert=True, kind="table")

        # fast case — we have a vocabulary and no fields
        if self._vocab is not None and not self._fields:
            return ItemList(
                item_nums=self.numbers()[mask], vocabulary=self.vocabulary, ordered=self.ordered
            )

        return self._take(mask)

    def _take(self, sel: ILIndexer, *, ordered: bool | None = None) -> ItemList:
        """
        Implementation helper for indexing.
        """
        indexer = get_indexer(sel)

        if ordered is None:
            ordered = self.ordered

        # Only subset the IDs if we don't have a vocabulary.  Otherwise, defer
        # ID subset until IDs are actually needed.
        array = self.to_arrow(ids=self.vocabulary is None, numbers=True, type="array")
        array = indexer(array)

        return ItemList(_init_array=array, vocabulary=self.vocabulary, ordered=ordered)  # type: ignore

    def __len__(self):
        return self._len

    def __getitem__(self, sel: ILIndexer) -> ItemList:
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
        return self._take(sel)

    def __getstate__(self) -> dict[str, object]:
        state: dict[str, object] = {"ordered": self.ordered, "len": self._len}
        if self._ids is not None:
            state["ids"] = self._ids
        elif self._vocab is not None:
            # compute the IDs so we can save them
            state["ids"] = self.ids(format="arrow")

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

    def __str__(self) -> str:  # pragma: nocover
        return f"<ItemList of {self._len} items>"

    def __repr__(self) -> str:  # pragma: nocover
        out = io.StringIO()
        nf = len(self._fields)
        print(f"<ItemList of {self._len} items with {nf} fields", "{", file=out)

        if self._numbers is not None:
            print("  numbers:", np.array2string(self._numbers.numpy(), threshold=10), file=out)
        if self._ids is not None:
            print("  ids:", np.array2string(self.ids(), threshold=10), file=out)
        for name, f in self._fields.items():
            print(f"  {name}:", np.array2string(f.numpy(), threshold=10), file=out)
        print("}>", end="", file=out)

        return out.getvalue()
