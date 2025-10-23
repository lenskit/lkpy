# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import warnings

from lenskit.data import ItemList

from .._base import DecomposedMetric, GlobalMetric, ListMetric, Metric

__all__ = ["Metric", "ListMetric", "GlobalMetric", "DecomposedMetric", "RankingMetricBase"]


class RankingMetricBase(Metric):
    """
    Base class for most ranking metrics, implementing an ``n`` parameter for
    truncation.

    Args:
        n:
            Specify the length cutoff for rankings. Rankings longer than this
            will be truncated prior to measurement.
        k:
            Deprecated alias for ``n``.

    Stability:
        Caller
    """

    n: int | None = None
    "The maximum length of rankings to consider."

    def __init__(self, n: int | None = None, *, k: int | None = None):
        if n is None:
            if k is not None:
                warnings.warn("k= is deprecated, use n=", DeprecationWarning)
                n = k

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
        Default name — class name, optionally @K.
        """
        name = self.__class__.__name__
        if self.n is not None:
            return f"{name}@{self.n}"
        else:
            return name

    def truncate(self, items: ItemList):
        """
        Truncate an item list if it is longer than :attr:`k`.
        """
        if self.n is not None:
            if not items.ordered:
                raise ValueError("top-k filtering requires ordered list")
            if len(items) > self.n:
                return items[: self.n]

        return items
