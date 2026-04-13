# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.data import ItemList

from .._base import ListMetric, Metric

__all__ = ["Metric", "ListMetric", "RankingMetricBase"]


class RankingMetricBase(Metric):
    """
    Base class for most ranking metrics, implementing an ``n`` parameter for
    truncation.

    .. versionchanged:: 2026.1

        Removed deprecated ``k`` alias for ``n``.

    Args:
        n:
            Specify the length cutoff for rankings. Rankings longer than this
            will be truncated prior to measurement.

    Stability:
        Caller
    """

    n: int | None = None
    "The maximum length of rankings to consider."

    def __init__(self, n: int | None = None):
        if n is not None and n < 0:
            raise ValueError("n must be positive or None")
        self.n = n

    @property
    def k(self):
        return self.n

    @k.setter
    def set_k(self, k, /):
        self.n = k

    @property
    def label(self):
        """
        Default name — class name, optionally @N.
        """
        name = self.__class__.__name__
        if self.n is not None:
            return f"{name}@{self.n}"
        else:
            return name

    def truncate(self, items: ItemList):
        """
        Truncate an item list if it is longer than :attr:`n`.
        """
        if self.n is not None:
            if not items.ordered:
                raise ValueError("top-n filtering requires ordered list")
            if len(items) > self.n:
                return items[: self.n]

        return items
