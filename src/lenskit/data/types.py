# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Basic data types used in data representations.
"""

# pyright: strict
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import ArrayLike
from typing_extensions import Any, Generic, Literal, Sequence, TypeAlias, TypeVar

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
IDSequence: TypeAlias = (
    Sequence[ID]
    | IDArray
    | pa.StringArray
    | "pa.IntegerArray[Any]"
    | "pa.ChunkedArray[Any]"
    | "pd.Series[CoreID]"
)
"Sequences of identifiers."

V = TypeVar("V", bound=np.number[Any], default=np.float32)
NPMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[V]]
NPVector: TypeAlias = np.ndarray[tuple[int], np.dtype[V]]

T = TypeVar("T")

DF_FORMAT: TypeAlias = Literal["numpy", "pandas", "torch"]
MAT_FORMAT: TypeAlias = Literal["scipy", "torch", "pandas", "structure"]
MAT_AGG: TypeAlias = Literal["count", "sum", "mean", "first", "last"]
LAYOUT: TypeAlias = Literal["csr", "coo"]


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


def argtopn(xs: ArrayLike, n: int) -> NPVector[np.int64]:
    """
    Compute the ordered positions of the top *n* elements.  Similar to
    :func:`torch.topk`, but works with NumPy arrays and only returns the
    indices.

    .. deprecated:: 2025.3.0

        This was never declared stable, but is now deprecated and will be
        removed in 2026.1.
    """
    if n == 0:
        return np.empty(0, np.int64)

    xs = np.asarray(xs)

    N = len(xs)
    invalid = np.isnan(xs)
    if np.any(invalid):
        mask = ~invalid
        vxs = xs[mask]
        remap = np.arange(N)[mask]
        res = argtopn(vxs, n)
        return remap[res]

    if n >= 0 and n < N:
        parts = np.argpartition(-xs, n)
        top_scores = xs[parts[:n]]
        top_sort = np.argsort(-top_scores)
        order = parts[top_sort]
    else:
        order = np.argsort(-xs)

    return order
