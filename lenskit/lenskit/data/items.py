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

from lenskit.types import EntityId, NPEntityId

from .checks import check_1d
from .mtarray import MTArray, MTGenericArray
from .vocab import Vocabulary

Backend: TypeAlias = Literal["numpy", "torch"]


class ItemList:
    """
    Representation of a (usually ordered) list of items, possibly with scores
    and other associated data; many components take and return item lists.  Item
    lists are to be treated as **immutable** — create a new list with modified
    data, do not do in-place modifications of the list itself or the arrays or
    data frame it returns.

    An item list logically a list of rows, each of which is an item, like a
    :class:`~pandas.DataFrame` but supporting multiple array backends.

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
            An array of scores for the items.
        fields:
            Additional fields, such as ``score`` or ``rating``.  Field names
            should generally be singular; the named keyword arguments and
            accessor methods are plural for readability (“get the list of item
            IDs”)
    """

    ordered: bool
    "Whether this list has a meaningful order."
    _len: int
    _ids: np.ndarray[int, np.dtype[NPEntityId]] | None = None
    _numbers: MTArray[np.int32] | None = None
    _vocab: Vocabulary | None = None
    _ranks: MTArray[np.int32] | None = None
    _fields: dict[str, MTGenericArray]

    def __init__(
        self,
        *,
        item_ids: NDArray[NPEntityId] | pd.Series[EntityId] | Sequence[EntityId] | None = None,
        item_nums: NDArray[np.int32] | pd.Series[int] | Sequence[int] | ArrayLike | None = None,
        vocabulary: Vocabulary | None = None,
        ordered: bool = False,
        scores: NDArray[np.generic] | torch.Tensor | ArrayLike | None = None,
        **fields: NDArray[np.generic] | torch.Tensor | ArrayLike,
    ):
        self.ordered = ordered
        self._vocab = vocabulary

        if item_ids is None and "item_id" in fields:
            item_ids = np.asarray(cast(Any, fields["item_id"]))

        if item_nums is None and "item_num" in fields:
            item_nums = np.asarray(cast(Any, fields["item_num"]))
            if not issubclass(item_nums.dtype.type, np.integer):
                raise TypeError("item numbers not integers")

        if item_ids is None and item_nums is None:
            self._ids = np.ndarray(0, dtype=np.int32)
            self._numbers = MTArray(np.ndarray(0, dtype=np.int32))
            self._len = 0

        if item_ids is not None:
            self._ids = np.asarray(item_ids)
            if not issubclass(self._ids.dtype.type, (np.integer, np.str_, np.bytes_, np.object_)):
                raise TypeError(f"item IDs not integers or bytes (type: {self._ids.dtype})")

            check_1d(self._ids, label="item_ids")
            self._len = len(item_ids)

        if item_nums is not None:
            self._numbers = MTArray(item_nums)
            check_1d(self._numbers, getattr(self, "_len", None), label="item_nums")
            self._len = self._numbers.shape[0]

        # convert fields and drop singular ID/number aliases
        self._fields = {
            name: check_1d(MTArray(data), self._len, label=name)
            for (name, data) in fields.items()
            if name not in ("item_id", "item_num")
        }

        if scores is not None:
            if "score" in fields:  # pragma: nocover
                raise ValueError("cannot specify both scores= and score=")
            self._fields["score"] = MTArray(scores)

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, *, vocabulary=Vocabulary, keep_user: bool = False
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
        df = df.drop(columns=to_drop, errors="ignore")

        fields = {f: df[f].values for f in df.columns}
        return cls(item_ids=ids, item_nums=nums, vocabulary=vocabulary, **fields)  # type: ignore

    def clone(self) -> ItemList:
        """
        Make a shallow copy of the item list.
        """
        return ItemList(
            item_ids=self._ids,
            item_nums=self._numbers,
            vocabulary=self._vocab,
            ordered=self.ordered,
            **self._fields,
        )

    def ids(self) -> NDArray[NPEntityId]:
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
            self._ids = cast(NDArray[NPEntityId], self._vocab.ids(self._numbers.numpy()))

        return self._ids

    @overload
    def numbers(
        self, format: Literal["numpy"] = "numpy", *, vocabulary: Vocabulary | None = None
    ) -> NDArray[np.int32]: ...
    @overload
    def numbers(
        self, format: Literal["torch"], *, vocabulary: Vocabulary | None = None
    ) -> torch.Tensor: ...
    @overload
    def numbers(
        self, format: LiteralString = "numpy", *, vocabulary: Vocabulary | None = None
    ) -> ArrayLike: ...
    def numbers(
        self, format: LiteralString = "numpy", *, vocabulary: Vocabulary | None = None
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
        if vocabulary is not None and vocabulary is not self._vocab:
            # we need to translate vocabulary
            ids = self.ids()
            return vocabulary.numbers(ids)

        if self._numbers is None:
            if self._vocab is None:
                raise RuntimeError("item numbers not available (no IDs or vocabulary provided)")
            assert self._ids is not None
            self._numbers = MTArray(self._vocab.numbers(self._ids))

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
        if ids and self._ids is not None or self._vocab is not None:
            cols["item_id"] = self.ids()
        if numbers and self._numbers is not None or self._vocab is not None:
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

    def __len__(self):
        return self._len

    def __getitem__(
        self,
        sel: NDArray[np.bool_] | NDArray[np.integer] | Sequence[int] | torch.Tensor | int | slice,
    ) -> ItemList:
        """
        Subset the item list.

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
            item_ids=iids, item_nums=nums, vocabulary=self._vocab, ordered=self.ordered, **flds
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
