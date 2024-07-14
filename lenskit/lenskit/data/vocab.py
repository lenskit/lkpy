"""
Vocabularies of IDs, tags, etc.
"""

from typing import Generic, Hashable, Sequence, TypeAlias, TypeVar

import numpy as np
import pandas as pd

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

    def __init__(self, keys: pd.Index | Sequence[VT] | None = None):
        if keys is None:
            keys = pd.Index()
        elif isinstance(keys, pd.Index):
            if not pd.unique:
                raise ValueError("vocabulary index must be unique")
        else:
            keys = pd.Index(np.unique(keys))  # type: ignore

        self._index = keys

    @property
    def index(self) -> pd.Index:
        "The property as a Pandas index."
        return self._index

    @property
    def size(self) -> int:
        "Current vocabulary size."
        return len(self._index)

    def __len__(self) -> int:
        return self.size
