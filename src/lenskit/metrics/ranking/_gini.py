# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from typing_extensions import override

from lenskit.data import Dataset, ItemList
from lenskit.logging import get_logger
from lenskit.stats import gini

from ._base import DecomposedMetric, RankingMetricBase
from ._weighting import GeometricRankWeight, RankWeight

_log = get_logger(__name__)


class GiniBase(DecomposedMetric, RankingMetricBase):
    """
    Base class for Gini diversity / popularity concentration metrics.
    """

    item_count: int

    def __init__(
        self,
        n: int | None = None,
        *,
        k: int | None = None,
        items: int | pd.Series | pd.DataFrame | Dataset,
    ):
        super().__init__(n, k=k)
        if isinstance(items, int):
            self.item_count = items
        elif isinstance(items, Dataset):
            self.item_count = items.item_count
        else:
            self.item_count = len(items)


class ListGini(GiniBase):
    """
    Measure item diversity of recommendations with the Gini coefficient.

    This computes the Gini coefficient of the *number of lists* that each item
    appears in.

    Args:
        n:
            The maximum recommendation list length.
        items:
            The total number of items, a data frame or series of item data, or a
            dataset. If a frame or series is provided, its length will be used
            as the number of items.  If a dataset is provided, its item count
            will be used.

    Stability:
        Caller
    """

    @override
    def compute_list_data(self, output: ItemList, test):
        recs = self.truncate(output)
        return recs.ids(format="arrow")

    @override
    def global_aggregate(self, values: list[pa.Array]):
        log = _log.bind(metric=self.label, item_count=self.item_count)
        log.debug("aggregating for %d lists", len(values))
        chunked = pa.chunked_array(values)
        vc_tbl = pc.value_counts(chunked)
        log.debug("found %d distinct items", len(vc_tbl))
        counts = np.zeros(self.item_count, np.int32)
        counts[: len(vc_tbl)] = np.asarray(vc_tbl.field("counts"), dtype=np.int32)
        return gini(counts)


class ExposureGini(GiniBase):
    """
    Measure exposure distribution of recommendations with the Gini coefficient.

    This uses a weighting model to compute the exposure of each item in each list,
    and computes the Gini coefficient of the total exposure.

    Args:
        n:
            The maximum recommendation list length.
        items:
            The total number of items, a data frame or series of item data, or a
            dataset. If a frame or series is provided, its length will be used
            as the number of items.  If a dataset is provided, its item count
            will be used.
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
        items: int | pd.Series | pd.DataFrame | Dataset,
        weight: RankWeight = GeometricRankWeight(),
    ):
        super().__init__(n=n, k=k, items=items)
        self.weight = weight

    @override
    def compute_list_data(self, output: ItemList, test):
        recs = self.truncate(output)
        weights = self.weight.weight(np.arange(1, len(recs) + 1))
        return (recs.ids(format="arrow"), pa.array(weights, type=pa.float32()))

    @override
    def global_aggregate(self, values: list[tuple[pa.Array, pa.FloatArray]]):
        log = _log.bind(metric=self.label, item_count=self.item_count)
        log.debug("aggregating for %d lists", len(values))
        table = pa.Table.from_batches(
            pa.RecordBatch.from_arrays([iv, ev], ["item_id", "exposure"]) for (iv, ev) in values
        )
        exp_tbl = table.group_by("item_id").aggregate([("exposure", "sum")])
        log.debug("found %d distinct items", exp_tbl.num_rows)
        exp = np.zeros(self.item_count, np.float32)
        exp[: exp_tbl.num_rows] = np.asarray(exp_tbl.column("exposure_sum"), dtype=np.float32)
        return gini(exp)
