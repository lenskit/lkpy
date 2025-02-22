# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Basic set statistics.
"""

from lenskit.data.items import ItemList

from ._base import Metric


class ListLength(Metric):
    """
    Report the length of the output (recommendation list or predictions).

    Stability:
        Caller
    """

    label = "N"  # type: ignore

    def __call__(self, recs: ItemList, test: ItemList) -> float:
        return len(recs)


class TestItemCount(Metric):
    """
    Report the number of test items.

    Stability:
        Caller
    """

    label = "TestItemCount"  # type: ignore

    def __call__(self, recs: ItemList, test: ItemList) -> float:
        return len(test)
