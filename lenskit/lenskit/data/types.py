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
from typing import Any, Generic, Literal, Sequence, TypeAlias, TypeVar, cast

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
            return UIPair(user=value, item=value)
