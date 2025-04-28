# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.data import ItemList

from .._base import GlobalMetric, ListMetric, Metric

__all__ = ["Metric", "ListMetric", "GlobalMetric", "RankingMetricBase"]


class RankingMetricBase(Metric):
    """
    Base class for most ranking metrics, implementing a ``k`` parameter for
    truncation.

    Args:
        k:
            Specify the length cutoff for rankings. Rankings longer than this
            will be truncated prior to measurement.

    Stability:
        Caller
    """

    k: int | None = None
    "The maximum length of rankings to consider."

    def __init__(self, k: int | None = None):
        if k is not None and k < 0:
            raise ValueError("k must be positive or None")
        self.k = k

    @property
    def label(self):
        """
        Default name — class name, optionally @K.
        """
        name = self.__class__.__name__
        if self.k is not None:
            return f"{name}@{self.k}"
        else:
            return name

    def truncate(self, items: ItemList):
        """
        Truncate an item list if it is longer than :attr:`k`.
        """
        if self.k is not None:
            if not items.ordered:
                raise ValueError("top-k filtering requires ordered list")
            if len(items) > self.k:
                return items[: self.k]

        return items
