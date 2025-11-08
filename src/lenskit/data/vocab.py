# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Vocabularies of IDs, tags, etc.
"""

# pyright: basic
from __future__ import annotations

from typing import Hashable, Iterable, Iterator, Literal, Sequence, overload

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import ArrayLike, NDArray
from structlog.stdlib import BoundLogger

from lenskit import _accel
from lenskit.logging import get_logger, trace

from .types import ID, IDArray, IDSequence

_log = get_logger(__name__)


class Vocabulary:
    """
    Vocabularies of entity identifiers for the LensKit data model.

    This class supports bidirectional mappings between key-like data and
    congiguous nonnegative integer indices.  Its key use is to facilitate the
    entity ID vocabularies in :class:`~lenskit.data.Dataset`, but it can also be
    used for things like item tags.

    IDs in a vocabulary must be unique.  Constructing a vocabulary with
    ``reorder=True`` ensures uniqueness (and sorts the IDs), but does not
    preserve the order of IDs in the original input.

    It is currently a wrapper around :class:`pandas.Index`, but this fact is not
    part of the stable public API.

    Args:
        keys:
            The IDs to put in the vocabulary.
        name:
            The vocabulary name (i.e. the entity class it stores IDs for).
        reorder:
            If ``True``, sort and deduplicate the IDs.  If ``False`` (the
            default), use the IDs as-is (assigning each to their position in the
            input sequence).

    Stability:
        Caller
    """

    name: str | None
    "The name of the vocabulary (e.g. “user”, “item”)."
    _pd_index: pd.Index | None = None
    "The Pandas index implementing the vocabulary."
    _array: pa.Array
    "PyArrow array for the vocabulary."
    _index: _accel.data.IDIndex
    "Internal index."
    _hash: str | None
    "Checksum of index data for fast equivalence testing."
    _log: BoundLogger

    def __init__(
        self,
        keys: IDSequence | pd.Index | Iterable[ID] | None = None,
        name: str | None = None,
        *,
        reorder: bool = True,
    ):
        self.name = name
        self._log = _log.bind(entity=name)
        key_arr: pa.Array
        if keys is None:
            key_arr = pa.array([], type=pa.int32())
        elif isinstance(keys, pd.Index):
            self._pd_index = keys
            key_arr = pa.array(keys.values)
        elif isinstance(keys, np.ndarray) or isinstance(keys, list) or isinstance(keys, pd.Series):
            key_arr = pa.array(keys)  # type: ignore
        elif isinstance(keys, pa.ChunkedArray):
            key_arr = keys.combine_chunks()
        elif isinstance(keys, pa.Array):
            key_arr = keys
        else:
            key_arr = pa.array(keys)  # type: ignore

        if reorder:
            self._pd_index = None
            key_arr = key_arr.drop_null().unique().sort()

        assert key_arr.null_count == 0
        self._array = key_arr
        self._index = _accel.data.IDIndex(key_arr)
        self._ensure_hash()

    def _ensure_hash(self):
        if not hasattr(self, "_hash"):
            # since we just made this array, we can assume the buffer is fully used
            self._hash = _accel.data.hash_array(self._array)

    @property
    def index(self) -> pd.Index:
        """
        The vocabulary as a Pandas index.

        Stability:
            Internal
        """
        if self._pd_index is None:
            self._pd_index = pd.Index(self._array.to_numpy(zero_copy_only=False))
            self._pd_index.name = f"{self.name}_id"
        return self._pd_index

    @property
    def size(self) -> int:
        "Current vocabulary size."
        return len(self._array)

    @overload
    def number(self, term: object, missing: Literal["error"] = "error") -> int: ...
    @overload
    def number(self, term: object, missing: Literal["none"] | None) -> int | None: ...
    def number(
        self, term: object, missing: Literal["error", "none"] | None = "error"
    ) -> int | None:
        "Look up the number for a vocabulary ID."
        idx = self._index.get_index(term)  # type: ignore
        if idx is None and missing == "error":
            raise KeyError(f"{self.name} ID {term}")

        return idx

    @overload
    def numbers(
        self,
        terms: Sequence[Hashable] | ArrayLike,
        missing: Literal["error", "negative"] = "error",
        *,
        format: Literal["numpy"] = "numpy",
    ) -> NDArray[np.int32]: ...
    @overload
    def numbers(
        self,
        terms: Sequence[Hashable] | ArrayLike,
        missing: Literal["error", "negative", "null"] = "error",
        *,
        format: Literal["arrow"],
    ) -> pa.Int32Array: ...
    def numbers(
        self,
        terms: Sequence[Hashable] | ArrayLike,
        missing: Literal["error", "negative", "null"] = "error",
        *,
        format: Literal["numpy", "arrow"] = "numpy",
    ) -> np.ndarray[tuple[int], np.dtype[np.int32]] | pa.Int32Array:
        "Look up the numbers for an array of terms or IDs."
        if pa.types.is_null(self._array.type):
            nums = pa.nulls(len(terms), type=pa.int32())
        else:
            term_arr = pa.array(terms, type=self._array.type)  # type: ignore
            nums = self._index.get_indexes(term_arr)  # type: ignore

        trace(self._log, "resolved %d IDs, %d invalid", len(terms), nums.null_count)
        if missing == "error" and nums.null_count:
            raise KeyError(f"{nums.null_count} invalid keys")
        elif missing == "negative":
            nums = pc.fill_null(nums, -1)  # type: ignore

        match format:
            case "numpy":
                return nums.to_numpy()
            case "arrow":
                return nums
            case _:  # pragma: nocover
                raise ValueError(f"invalid format {format}")

    def term(self, num: int) -> object:
        """
        Look up the term with a particular number.  Negative indexing is **not** supported.
        """
        if num < 0:
            raise IndexError("negative numbers not supported")
        return self._array[num].as_py()

    @overload
    def terms(
        self,
        nums: list[int] | NDArray[np.integer] | pd.Series | None = None,
        *,
        format: Literal["numpy"] = "numpy",
    ) -> IDArray: ...
    @overload
    def terms(
        self,
        nums: list[int] | NDArray[np.integer] | pd.Series | None = None,
        *,
        format: Literal["arrow"],
    ) -> pa.Array: ...
    def terms(
        self,
        nums: list[int] | NDArray[np.integer] | pd.Series | None = None,
        *,
        format: Literal["numpy", "arrow"] = "numpy",
    ) -> IDArray | pa.Array:
        """
        Get a list of terms, optionally for an array of term numbers.

        Args:
            nums:
                The numbers (indices) for of terms to retrieve.  If ``None``,
                returns all terms.

        Returns:
            The terms corresponding to the specified numbers, or the full array
            of terms (in order) if ``nums=None``.
        """
        ids = self._array
        if nums is not None:
            num_arr = pa.array(nums, type=pa.uint32())
            ids = ids.take(num_arr)

        match format:
            case "arrow":
                return ids
            case "numpy":
                return ids.to_numpy(zero_copy_only=False)
            case _:  # pragma: nocover
                raise ValueError(f"invalid format {format}")

    def id(self, num: int) -> object:
        "Alias for :meth:`term`  for greater readability for entity ID vocabularies."
        return self.term(num)

    @overload
    def ids(
        self,
        nums: list[int] | NDArray[np.integer] | pd.Series | None = None,
        *,
        format: Literal["numpy"] = "numpy",
    ) -> IDArray: ...
    @overload
    def ids(
        self,
        nums: list[int] | NDArray[np.integer] | pd.Series | None = None,
        *,
        format: Literal["arrow"],
    ) -> pa.Array: ...
    def ids(
        self,
        nums: list[int] | NDArray[np.integer] | pd.Series | None = None,
        *,
        format: Literal["numpy", "arrow"] = "numpy",
    ) -> IDArray | pa.Array:
        "Alias for :meth:`terms` for greater readability for entity ID vocabularies."
        return self.terms(nums, format=format)

    def id_array(self) -> pa.Array:
        return self._array

    def __eq__(self, other: Vocabulary) -> bool:  # noqa: F821
        if self is other:
            return True
        elif not isinstance(other, Vocabulary):
            return False
        else:
            self._ensure_hash()
            other._ensure_hash()
            return self._hash == other._hash

    def __contains__(self, key: object) -> bool:
        return self._index.get_index(key) is not None  # type: ignore

    def __iter__(self) -> Iterator[object]:
        return (t.as_py() for t in self._array)

    def __len__(self) -> int:
        return self.size

    def __array__(self, dtype=None) -> np.ndarray:
        return self._array.to_numpy(zero_copy_only=False)

    def __getstate__(self):
        return {"name": self.name, "array": self._array}

    def __setstate__(self, state):
        self.name = state["name"]
        self._array = state["array"]
        self._index = _accel.data.IDIndex(self._array)
        self._log = _log.bind(entity=self.name)

    def __str__(self) -> str:
        if self.name:
            return f"Vocabulary of {self.size} {self.name} terms"
        else:
            return f"Vocabulary of {self.size} terms"

    def __repr__(self) -> str:
        if self.name:
            return f"<Vocabulary(name={self.name}, size={self.size})>"
        else:
            return f"<Vocabulary(size={self.size})>"
