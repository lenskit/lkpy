# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Sequence

from lenskit.data import GenericKey, ItemListCollection


class BatchResults:
    """
    Results from a batch recommendation run.  Results consist of the outputs of
    various pipeline components for each of the test users.  Results may be
    ``None``, if the pipeline produced no output for that query.

    Stability:
        Caller
    """

    _key_schema: type[tuple] | Sequence[str]
    _data: dict[str, ItemListCollection[GenericKey]]

    def __init__(self, key: type[tuple] | Sequence[str]):
        """
        Construct a new set of batch results.
        """
        self._key_schema = key
        self._data = {}

    @property
    def outputs(self) -> list[str]:
        """
        Get the list of output names in these results.
        """
        return list(self._data.keys())

    def output(self, name: str) -> ItemListCollection[GenericKey]:
        """
        Get the item lists for a particular output component.

        Args:
            name:
                The output name. This may or may not be the same as the
                component name.
        """
        return self._data[name]

    def add_result(self, name: str, key: GenericKey, result: object):
        """
        Add a single result for one of the outputs.

        Args:
            name:
                The output name in which to save this result.
            user:
                The user identifier for this result.
            result:
                The result object to save.
        """

        if name not in self._data:
            self._data[name] = ItemListCollection(self._key_schema)

        try:
            self._data[name].add(result, *key)
        except TypeError as e:
            raise TypeError(f"invalid key {key} (for type {self._data[name].key_type})", e)
