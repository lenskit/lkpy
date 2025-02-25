# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Entity accessors for data sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
from typing_extensions import Any, overload

from lenskit.logging import get_logger

from .attributes import AttributeSet, attr_set
from .schema import EntitySchema
from .types import IDArray, IDSequence
from .vocab import Vocabulary

_log = get_logger(__name__)


class EntitySet:
    """
    Representation of a set of entities from the dataset.  Obtained from
    :meth:`Dataset.entities`.
    """

    name: str
    """
    The name of the entity class for these entities.
    """
    schema: EntitySchema
    vocabulary: Vocabulary
    """
    The identifier vocabulary for this schema.
    """
    _table: pa.Table
    """
    The Arrow table of entity information.
    """
    _selected: pa.Int32Array | None = None

    def __init__(
        self,
        name: str,
        schema: EntitySchema,
        vocabulary: Vocabulary,
        table: pa.Table,
        _sel: pa.Int32Array | None = None,
    ):
        self.name = name
        self.schema = schema
        self.vocabulary = vocabulary
        self._table = table
        self._selected = _sel

    @property
    def attributes(self) -> list[str]:
        return list(self.schema.attributes.keys())

    def count(self) -> int:
        """
        Return the number of entities in this entity set.
        """
        if self._selected is not None:
            return len(self._selected)
        else:
            return self._table.num_rows

    def ids(self) -> IDArray:
        """
        Get the identifiers of the entities in this set.  This is returned
        directly as PyArrow array instead of NumPy.
        """
        if self._selected is not None:
            return self.vocabulary.ids(self._selected.to_numpy())
        else:
            return self.vocabulary.ids()

    def numbers(self) -> np.ndarray[int, np.dtype[np.int32]]:
        """
        Get the numbers (from the vocabulary) for the entities in this set.
        """
        if self._selected is not None:
            return self._selected.to_numpy()
        else:
            return np.arange(self.count(), dtype=np.int32)

    def arrow(self) -> pa.Table:
        """
        Get these entities and their attributes as a PyArrow table.
        """
        if self._selected is not None:
            return self._table.take(self._selected)
        else:
            return self._table

    def pandas(self) -> pd.DataFrame:
        """
        Get the entities and their attributes as a Pandas data frame.
        """
        return self.arrow().to_pandas()

    def attribute(self, name: str) -> AttributeSet:
        """
        Get values of an attribute for the entites in this entity set.
        """
        spec = self.schema.attributes[name]

        return attr_set(name, spec, self._table, self.vocabulary, self._selected)

    @overload
    def select(self, *, ids: IDSequence | None = None) -> EntitySet: ...
    @overload
    def select(
        self,
        *,
        numbers: np.ndarray[int, np.dtype[np.integer[Any]]] | pa.IntegerArray[Any] | None = None,
    ) -> EntitySet: ...
    def select(
        self,
        *,
        ids: IDSequence | None = None,
        numbers: np.ndarray[int, np.dtype[np.integer[Any]]] | pa.IntegerArray[Any] | None = None,
    ) -> EntitySet:
        """
        Select a subset of the entities in this set.

        .. note::

            The vocabulary is unchanged, so numbers in the resulting set will be
            entity numbers in the dataset's vocabulary.  They are not rearranged
            to be relative to this entity set.

        Args:
            ids:
                The entity identifiers to select.
            numbers:
                The entity numbers to select.

        Returns:
            The entity subset.
        """
        if numbers is not None and ids is None:
            picked = pa.array(numbers).cast(pa.int32())
        elif ids is not None and numbers is None:
            picked = pa.array(self.vocabulary.numbers(ids)).cast(pa.int32())
        else:  # pragma: nocover
            raise ValueError("specify exactly one of ids and numbers")

        return EntitySet(self.name, self.schema, self.vocabulary, self._table, picked)  # type: ignore

    def __len__(self):
        return self.count()
