# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Per-user rating holdout methods for user-based data splitting.
"""

from typing import Protocol

import numpy as np

from lenskit.data import ItemList
from lenskit.random import RNGInput, random_generator


class HoldoutMethod(Protocol):
    """
    Holdout methods select test rows for a user (or occasionally an item).
    Partition methods are callable; when called with a data frame, they return
    the test entries.

    Stability:
        Caller
    """

    def __call__(self, items: ItemList) -> ItemList:
        """
        Subset an item list (in the uncommon case of item-based holdouts, the
        item list actually holds user IDs).

        Args:
            udf:
                The item list from which holdout items should be selected.

        Returns:
            The list of test items.
        """
        raise NotImplementedError()


class SampleN(HoldoutMethod):
    """
    Randomly select a fixed number of test rows per user/item.

    Stability:
        Caller

    Args:
        n:
            The number of test items to select.
        rng:
            The random number generator or seed (see :ref:`rng`).
    """

    n: int
    rng: np.random.Generator

    def __init__(self, n: int, rng: RNGInput = None):
        self.n = n
        self.rng = random_generator(rng)

    def __call__(self, items: ItemList) -> ItemList:
        if len(items) <= self.n:
            return items

        sel = self.rng.choice(len(items), self.n, replace=False)
        return items[sel]


class SampleFrac(HoldoutMethod):
    """
    Randomly select a fraction of test rows per user/item.

    Stability:
        Caller

    Args:
        frac:
            The fraction items to select for testing.
        rng:
            The random number generator or seed (see :ref:`rng`).
    """

    fraction: float
    rng: np.random.Generator

    def __init__(self, frac: float, rng: RNGInput = None):
        self.fraction = frac
        self.rng = random_generator(rng)

    def __call__(self, items: ItemList) -> ItemList:
        n = round(len(items) * self.fraction)
        sel = self.rng.choice(len(items), n, replace=False)
        return items[sel]


class LastN(HoldoutMethod):
    """
    Select a fixed number of test rows per user/item, based on ordering by a
    field.

    Stability:
        Caller

    Args:
        n: The number of test items to select.
        field: The field to order by.
    """

    n: int
    field: str

    def __init__(self, n: int, field: str = "timestamp"):
        self.n = n
        self.field = field

    def __call__(self, items: ItemList) -> ItemList:
        if len(items) <= self.n:
            return items

        col = items.field(self.field)
        if col is None:
            raise TypeError(f"item list does not have ordering field {self.field}")
        ordered = np.argsort(col)
        return items[ordered[-self.n :]]


class LastFrac(HoldoutMethod):
    """
    Select a fraction of test rows per user/item.

    Stability:
        Caller

    Args:
        frac(double): the fraction of items to select for testing.
    """

    fraction: float
    field: str

    def __init__(self, frac: float, field: str = "timestamp"):
        self.fraction = frac
        self.field = field

    def __call__(self, items: ItemList) -> ItemList:
        n = round(len(items) * self.fraction)

        col = items.field(self.field)
        if col is None:
            raise TypeError(f"item list does not have ordering field {self.field}")
        ordered = np.argsort(col)
        return items[ordered[-n:]]
