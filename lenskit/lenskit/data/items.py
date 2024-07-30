# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Primary item-list abstraction.
"""

from __future__ import annotations

from typing import Literal, LiteralString, Sequence, TypeAlias, TypeVar, overload

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike, NDArray

from lenskit.data.mtarray import MTArray, MTGenericArray
from lenskit.data.vocab import EntityId, NPEntityId, Vocabulary

Backend: TypeAlias = Literal["numpy", "torch"]
EID = TypeVar("EID", bound=EntityId)


class ItemList:
    """
    Representation of a (usually ordered) list of items, possibly with scores
    and other associated data.

    Args:
        item_ids:
            A list or array of item identifiers.
        item_nums:
            A list or array of item numbers.
        vocabulary:
            A vocabulary to translate between item IDs and numbers.
        fields:
            Additional fields, such as ``score`` or ``rating``.
    """

    _len: int
    _ids: np.ndarray[int, np.dtype[NPEntityId]] | None = None
    _numbers: MTArray[np.int32] | None = None
    _vocab: Vocabulary[EntityId] | None = None
    _fields: dict[str, MTGenericArray]

    def __init__(
        self,
        *,
        item_ids: NDArray[NPEntityId] | pd.Series[EntityId] | Sequence[EntityId] | None = None,
        item_nums: NDArray[np.int32] | pd.Series[int] | Sequence[int] | ArrayLike | None = None,
        vocabulary: Vocabulary[EID] | None = None,
        **fields: NDArray[np.generic] | torch.Tensor | ArrayLike,
    ):
        self._vocab = vocabulary

        if item_ids is None and item_nums is None:
            self._ids = np.ndarray(0, dtype=np.int32)
            self._numbers = MTArray(np.ndarray(0, dtype=np.int32))
            self._len = 0

        if item_ids is not None:
            self._ids = np.asarray(item_ids)
            if len(self._ids.shape) > 1:
                raise TypeError("item lists must be 1-dimensional")
            self._len = len(item_ids)

        if item_nums is not None:
            self._numbers = MTArray(item_nums)
            if hasattr(self, "_len"):
                if self._numbers.shape != (self._len,):
                    nl = self._numbers.shape[0]
                    raise ValueError(
                        f"item ID and number lists have different lengths ({self._len} != {nl})"
                    )
            else:
                self._len = self._numbers.shape[0]

        self._fields = {name: MTArray(data) for (name, data) in fields.items()}

    def clone(self) -> ItemList:
        """
        Make a shallow copy of the item list.
        """
        return ItemList(item_ids=self._ids, item_nums=self._numbers, vocabulary=self._vocab)

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
            self._ids = self._vocab.ids(self._numbers.numpy())

        return self._ids

    @overload
    def numbers(self, format: Literal["numpy"] = "numpy") -> NDArray[np.int32]: ...
    @overload
    def numbers(self, format: Literal["torch"]) -> torch.Tensor: ...
    @overload
    def numbers(self, format: LiteralString = "numpy") -> ArrayLike: ...
    def numbers(self, format: LiteralString = "numpy") -> ArrayLike:
        """
        Get the item numbers.

        Args:
            format:
                The array format to use.

        Returns:
            An array of item numbers.

        Raises:
            RuntimeError: if the item list was not created with numbers or a :class:`Vocabulary`.
        """
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
    def scores(self, format: LiteralString = "numpy") -> ArrayLike | None: ...
    def scores(self, format: LiteralString = "numpy") -> ArrayLike | None:
        """
        Get the item scores (if available).
        """
        return self.field("scores", format)

    @overload
    def field(
        self, name: str, format: Literal["numpy"] = "numpy"
    ) -> NDArray[np.floating] | None: ...
    @overload
    def field(self, name: str, format: Literal["torch"]) -> torch.Tensor | None: ...
    @overload
    def field(self, name: str, format: LiteralString) -> ArrayLike | None: ...
    def field(self, name: str, format: LiteralString = "numpy") -> ArrayLike | None:
        val = self._fields.get(name, None)
        if val is None:
            return None
        else:
            return val.to(format)

    def __len__(self):
        return self._len
