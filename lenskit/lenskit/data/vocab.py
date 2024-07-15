"""
Vocabularies of IDs, tags, etc.
"""

# pyright: basic
from typing import Generic, Hashable, Iterable, Literal, Sequence, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

EntityId: TypeAlias = int | str | bytes
"Allowable entity identifier types."

VT = TypeVar("VT", bound=Hashable)
"Term type in a vocabulary."


class Vocabulary(Generic[VT]):
    """
    Vocabularies of terms, tags, entity IDs, etc. for the LensKit data model.

    This class supports bidirectional mappings between key-like data and
    congiguous nonnegative integer indices.  Its key use is to facilitate the
    user and item ID vocabularies in :class:`~lenskit.data.Dataset`, but it can
    also be used for things like item tags.

    It is currently a wrapper around :class:`pandas.Index`, but supports the
    ability to add additional vocabulary terms after the vocabulary has been
    created.  New terms do not change the index positions of previously-known
    identifiers.
    """

    _index: pd.Index

    def __init__(self, keys: pd.Index | Iterable[VT] | None = None):
        if keys is None:
            keys = pd.Index()
        elif isinstance(keys, pd.Index):
            if not pd.unique:
                raise ValueError("vocabulary index must be unique")
        elif isinstance(keys, np.ndarray) or isinstance(keys, list) or isinstance(keys, pd.Series):
            keys = pd.Index(np.unique(keys))  # type: ignore
        else:
            keys = pd.Index(np.unique(list(set(keys))))  # type: ignore

        self._index = keys

    @property
    def index(self) -> pd.Index:
        "The property as a Pandas index."
        return self._index

    @property
    def size(self) -> int:
        "Current vocabulary size."
        return len(self._index)

    def number(self, term: VT, missing: Literal["error", "negative"] = "error") -> int:
        "Look up the number for a vocabulary term."
        try:
            num = self._index.get_loc(term)
            assert isinstance(num, int)
            return num
        except KeyError as e:
            if missing == "negative":
                return -1
            else:
                raise e

    def numbers(
        self, terms: Sequence[VT] | ArrayLike, missing: Literal["error", "negative"] = "error"
    ) -> np.ndarray[int, np.dtype[np.int32]]:
        "Look up the numbers for an array of terms or IDs."
        nums = np.require(self._index.get_indexer_for(terms), dtype=np.int32)
        if missing == "error" and np.any(nums < 0):
            raise KeyError()
        return nums

    def term(self, num: int) -> VT:
        """
        Look up the term with a particular number.  Negative indexing is **not** supported.
        """
        if num < 0:
            raise IndexError("negative numbers not supported")
        return self._index[num]

    def terms(self, nums: list[int] | NDArray[np.integer] | pd.Series) -> np.ndarray:
        "Look up the terms for an array of indices."
        nums = np.asarray(nums, dtype=np.int32)
        if np.any(nums < 0):
            raise IndexError("negative numbers not supported")
        return self._index[nums].values

    def id(self, num: int) -> VT:
        "Alias for :meth:`term`  for greater readability for entity ID vocabularies."
        return self.term(num)

    def ids(self, nums: list[int] | NDArray[np.integer] | pd.Series) -> np.ndarray:
        "Alias for :meth:`terms` for greater readability for entity ID vocabularies."
        return self.terms(nums)

    def add_terms(self, terms: list[VT] | ArrayLike):
        arr = np.unique(terms)  # type: ignore
        nums = self.numbers(arr, missing="negative")
        fresh = arr[nums < 0]
        self._index = pd.Index(np.concatenate([self._index.values, fresh]))

    def __len__(self) -> int:
        return self.size
