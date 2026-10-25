# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Basic set statistics.
"""

from typing import override

from lenskit.data import ID, ItemList
from lenskit.data.accum import Accumulator
from lenskit.data.types import IDSequence
from lenskit.metrics import Metric

from ._base import ListMetric


class ListLength(ListMetric):
    """
    Report the length of the output (recommendation list or predictions).

    Stability:
        Caller
    """

    label = "N"  # type: ignore

    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        return len(recs)


class TestItemCount(ListMetric):
    """
    Report the number of test items.

    Stability:
        Caller
    """

    label = "TestItemCount"  # type: ignore

    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        return len(test)


class UniqueItemCount(Metric[IDSequence, int]):
    """
    Count the number of unique items in the recommendation lists.

    Stability:
        Caller
    """

    def measure_list(self, recs: ItemList, test: ItemList) -> IDSequence:
        return recs.ids()

    def create_accumulator(self):
        return UniqueItemAccumulator()


class UniqueItemAccumulator(Accumulator):
    _ids: set[ID]

    def __init__(self):
        self._ids = set()

    @override
    def add(self, value: IDSequence) -> None:
        self._ids.update(value)

    @override
    def accumulate(self) -> int:
        return len(self._ids)
