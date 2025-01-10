# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Basic data types used in data representations.
"""

# pyright: strict
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, NamedTuple, Sequence, TypeAlias, TypedDict, TypeVar, cast

import numpy as np
import pandas as pd

FeedbackType: TypeAlias = Literal["explicit", "implicit"]
"Types of feedback supported."

CoreID: TypeAlias = int | str | bytes
"Core (non-NumPy) identifier types."
NPID: TypeAlias = np.integer[Any] | np.str_ | np.bytes_ | np.object_
"NumPy entity identifier types."
ID: TypeAlias = CoreID | NPID
"Allowable identifier types."
IDArray: TypeAlias = np.ndarray[tuple[int], np.dtype[NPID]]
"NumPy arrays of identifiers."
IDSequence: TypeAlias = Sequence[ID] | IDArray | "pd.Series[CoreID]"
"Sequences of identifiers."

T = TypeVar("T")


@dataclass
class AliasedColumn:
    """
    A data frame column with possible aliases.

    Stability:
        Testing
    """

    name: str
    "The column name."
    compat_aliases: list[str] = field(default_factory=list)
    "A list of aliases for the column."
    warn: bool = False
    "Whether to warn when using an alias."


Column: TypeAlias = str | AliasedColumn


@dataclass(frozen=True)
class UIPair(Generic[T]):
    """
    A user-item pair of values.
    """

    user: T
    item: T

    @classmethod
    def normalize(cls, value: T | UIPair[T] | tuple[T, T]) -> UIPair[T]:
        if isinstance(value, UIPair):
            return cast(UIPair[T], value)
        elif isinstance(value, tuple):
            user, item = cast(tuple[T, T], value)
            return cls(user=user, item=item)
        else:
            return UITuple(user=value, item=value)  # type: ignore


class UIDict(TypedDict, Generic[T]):
    user: T
    item: T


class UITuple(NamedTuple, Generic[T]):
    """
    Tuple of (user, item) data, typically for configuration and similar
    purposes.

    Stability:
        Caller
    """

    user: T
    "User data."
    item: T
    "Item data."

    @classmethod
    def create(cls, x: UITuple[T] | tuple[T, T] | UIDict[T] | T) -> UITuple[T]:
        """
        Create a user-item tuple from a tuple or data.  If a single value
        is provided, it is used for both user and item.
        """
        if isinstance(x, UITuple):
            return cast(UITuple[T], x)
        elif isinstance(x, (tuple, list)):
            u, i = cast(tuple[T, T], x)
            return UITuple(u, i)
        elif isinstance(x, dict):
            return UITuple(x["user"], x["item"])
        else:
            return UITuple(x, x)
