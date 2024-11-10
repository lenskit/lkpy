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

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, NamedTuple, TypeAlias, TypeVar, cast

import numpy as np

FeedbackType: TypeAlias = Literal["explicit", "implicit"]
"Types of feedback supported."

EntityId: TypeAlias = int | str | bytes
"Allowable entity identifier types."
NPEntityId: TypeAlias = np.integer[Any] | np.str_ | np.bytes_ | np.object_
"Allowable entity identifier types (NumPy version)"

T = TypeVar("T")


@dataclass
class AliasedColumn:
    name: str
    "The column name."
    compat_aliases: list[str] = field(default_factory=list)
    "A list of aliases for the column."
    warn: bool = False
    "Whether to warn when using an alias."


Column: TypeAlias = str | AliasedColumn


if TYPE_CHECKING or sys.version_info >= (3, 11):

    class UITuple(NamedTuple, Generic[T]):
        """
        Tuple of (user, item) data, typically for configuration and similar
        purposes.
        """

        user: T
        "User data."
        item: T
        "Item data."

        @classmethod
        def create(cls, x: UITuple[T] | tuple[T, T] | T) -> UITuple[T]:
            """
            Create a user-item tuple from a tuple or data.  If a single value
            is provided, it is used for both user and item.
            """
            if isinstance(x, UITuple):
                return cast(UITuple[T], x)
            elif isinstance(x, tuple):
                u, i = cast(tuple[T, T], x)
                return UITuple(u, i)
            else:
                return UITuple(x, x)
else:

    class UITuple(NamedTuple):
        """
        Tuple of (user, item) data, typically for configuration and similar
        purposes.
        """

        user: Any
        "User data."
        item: Any
        "Item data."

        @classmethod
        def create(cls, x: UITuple | tuple[Any, Any] | Any) -> UITuple:
            """
            Create a user-item tuple from a tuple or data.  If a single value
            is provided, it is used for both user and item.
            """
            if isinstance(x, UITuple):
                return x
            elif isinstance(x, tuple):
                u, i = x
                return UITuple(u, i)
            else:
                return UITuple(x, x)
