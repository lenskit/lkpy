# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from lenskit.data import Dataset, ItemList, Vocabulary
from lenskit.logging import get_logger
from lenskit.stats import gini

from ._base import RankingMetricBase
from ._weighting import GeometricRankWeight, RankWeight

_log = get_logger(__name__)


class GiniBase(RankingMetricBase):
    """
    Base class for Gini diversity / popularity concentration metrics.

    Args:
        n:
            The maximum recommendation list length.
        items:
            The item vocabulary or a dataset from which to extract the items.
    """

    item_vocab: Vocabulary

    def __init__(
        self,
        n: int | None = None,
        *,
        k: int | None = None,
        items: Vocabulary | Dataset,
    ):
        super().__init__(n, k=k)
        if isinstance(items, Dataset):
            self.item_vocab = items.items
        else:
            self.item_vocab = items

    def create_accumulator(self):
        return GiniAccumulator(len(self.item_vocab))


class ListGini(GiniBase):
    """
    Measure item diversity of recommendations with the Gini coefficient.

    This computes the Gini coefficient of the *number of lists* that each item
    appears in.

    Args:
        n:
            The maximum recommendation list length.
        items:
            The item vocabulary or a dataset from which to extract the items.

    Stability:
        Caller
    """

    @override
    def measure_list(self, output: ItemList, test) -> tuple[NDArray[np.int32], float]:
        recs = self.truncate(output)
        ids = recs.numbers(vocabulary=self.item_vocab)
        return (ids, 1.0)


class ExposureGini(GiniBase):
    """
    Measure exposure distribution of recommendations with the Gini coefficient.

    This uses a weighting model to compute the exposure of each item in each list,
    and computes the Gini coefficient of the total exposure.

    Args:
        n:
            The maximum recommendation list length.
        items:
            The item vocabulary or a dataset from which to extract the items.
        weight:
            The rank weighting model to use.  Defaults to
            :class:`GeometricRankWeight` with the specified patience parameter.

    Stability:
        Caller
    """

    weight: RankWeight

    def __init__(
        self,
        n: int | None = None,
        *,
        k: int | None = None,
        items: Vocabulary | Dataset,
        weight: RankWeight = GeometricRankWeight(),
    ):
        super().__init__(n=n, k=k, items=items)
        self.weight = weight

    @override
    def measure_list(self, output: ItemList, test) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        recs = self.truncate(output)
        ids = recs.numbers(vocabulary=self.item_vocab)
        weights = self.weight.weight(np.arange(1, len(recs) + 1, dtype=np.int32))
        return (ids, weights)


class GiniAccumulator:
    n_items: int
    totals: np.ndarray[tuple[int], np.dtype[np.float64]]

    def __init__(self, n: int):
        self.n_items = n
        self.totals = np.zeros(n)

    def add(self, value: tuple[NDArray[np.int32], NDArray[np.float32] | float]) -> None:
        items, values = value
        self.totals[items] += values

    def accumulate(self) -> float:
        dist = self.totals / self.totals.sum()
        return gini(dist)
