# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Basic data types used in data representations.
"""

# pyright: strict
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np
import pandas as pd
import pyarrow as pa
from typing_extensions import TypeVar

__all__ = [
    "DF_FORMAT",
    "ID",
    "LAYOUT",
    "MAT_FORMAT",
    "NPID",
    "AliasedColumn",
    "Extent",
    "FeedbackType",
    "IDArray",
    "IDSequence",
    "NPMatrix",
    "NPVector",
    "UIPair",
]

type FeedbackType = Literal["explicit", "implicit"]
"Types of feedback supported."

type CoreID = int | str | bytes
"Core (non-NumPy) identifier types."
type NPID = np.integer[Any] | np.str_ | np.bytes_ | np.object_
"NumPy entity identifier types."
type ID = CoreID | NPID
"Allowable identifier types."
type IDArray = np.ndarray[tuple[int], np.dtype[NPID]]
"NumPy arrays of identifiers."
type IDSequence = (
    Sequence[ID]
    | IDArray
    | pa.StringArray
    | pa.IntegerArray[Any]
    | pa.ChunkedArray[Any]
    | pd.Series[CoreID]
)
"Sequences of identifiers."

V = TypeVar("V", bound=np.number[Any], default=np.float32)
NPMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[V]]
NPVector: TypeAlias = np.ndarray[tuple[int], np.dtype[V]]

type DF_FORMAT = Literal["numpy", "pandas", "torch"]
type MAT_FORMAT = Literal["scipy", "torch", "pandas", "structure"]
type LAYOUT = Literal["csr", "coo"]


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


type Column = str | AliasedColumn


@dataclass(frozen=True)
class UIPair[T]:
    """
    A user-item pair of values.
    """

    user: T
    item: T


class Extent(NamedTuple):
    """
    Representation of a range with a start and end.
    """

    start: int
    "The range start (inclusive)."
    end: int
    "The range end (exclusive)."

    @property
    def size(self) -> int:
        """
        The size of the extent.
        """
        return self.end - self.start
