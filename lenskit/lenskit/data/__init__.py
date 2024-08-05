# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing_extensions import Generic, Literal, NamedTuple, TypeAlias, TypeVar

from .vocab import EntityId, Vocabulary  # noqa: F401, E402

FeedbackType: TypeAlias = Literal["explicit", "implicit"]
"Types of feedback supported."


T = TypeVar("T")


class UITuple(Generic[T], NamedTuple):
    """
    Tuple of (user, item) data, typically for configuration and similar
    purposes.
    """

    user: T
    item: T

    @classmethod
    def create(cls, x: UITuple[T] | tuple[T, T] | T) -> UITuple[T]:
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


from .dataset import Dataset, from_interactions_df  # noqa: F401, E402
from .items import ItemList  # noqa: F401, E402
from .movielens import load_movielens  # noqa: F401, E402
from .mtarray import MTArray, MTFloatArray, MTGenericArray, MTIntArray  # noqa: F401, E402
