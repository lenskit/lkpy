# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from lenskit.data import ItemList, ItemListCollection
from lenskit.data.accum import Accumulator
from lenskit.logging import get_logger, item_progress

from ._base import (
    FunctionMetric,
    Metric,
    MetricFunction,
    MetricResult,
    MetricVal,
)

_log = get_logger(__name__)


@dataclass
class MetricState:
    """
    Internal class for storing metrics with their configuration and
    accumulators.

    Stability:
        Internal
    """

    metric: Metric
    accumulator: Accumulator[Any, MetricResult]
    label: str


class MeasurementCollector:
    """
    Collect metric measurements over multiple recommendation lists.

    This class automates collecting metric values and translating accumulated
    summaries into data frames.

    .. versionchanged:: 2026.1

        ``metrics`` is no longer publicly exposed as a list of wrappers.

    Stability:
        Caller
    """

    _metrics: list[MetricState]
    _list_records: list[dict[str, float | int | object]]
    key_fields: list[str]
    """
    Columns naming the keys of measured lists.
    """

    def __init__(self):
        self._metrics = []
        self._list_records = []
        self.key_fields = []

    def empty_copy(self):
        """
        Create a copy of this measurement collector with no collected data.
        """
        copy = MeasurementCollector()
        copy._metrics = [
            replace(m, accumulator=m.metric.create_accumulator()) for m in self._metrics
        ]
        return copy

    def reset(self):
        """
        Remove all collected data from this collector.
        """
        self.key_fields = []
        self._list_records = []
        for state in self._metrics:
            state.accumulator = state.metric.create_accumulator()

    @property
    def metric_names(self) -> list[str]:
        """
        Get the list of metric names.
        """
        return [m.label for m in self._metrics]

    def add_metric(
        self,
        metric: Metric | MetricFunction | type[Metric],
        label: str | None = None,
    ):
        """
        Add a metric to this accumulator.

        Args:
            metric:
                The metric to add.
            label:
                The label to use for the metric's results. If unset, obtains
                from the metric.
        """
        wrapper = _wrap_metric(metric, label)
        self._metrics.append(wrapper)

    def measure_list(self, output: ItemList, test: ItemList, **keys: Any):
        """
        Measure a single list and accumulate the intermediate results.

        Args:
            output:
                The recommendation list to measure.
            test:
                The ground truth test data.
            **keys:
                Identifying keys for this list (e.g., user_id).
        """
        if not self.key_fields:
            self.key_fields = list(keys.keys())

        rec = dict(keys)
        for state in self._metrics:
            intermediate = state.metric.measure_list(output, test)
            state.accumulator.add(intermediate)
            lv = state.metric.extract_list_metrics(intermediate)
            _add_values(rec, state.label, lv)

        self._list_records.append(rec)

    def measure_collection(
        self, outputs: ItemListCollection, test: ItemListCollection, **keys: Any
    ):
        """
        Measure a collection of item lists against truth data.

        Args:
            outputs:
                The item lists to measure.
            test:
                Test data item lists.
            keys:
                Additional keys to label measurements from these lists.
        """

        _log.debug("measuring %d metrics for %d output lists", len(self._metrics), len(outputs))
        no_test_count = 0
        with item_progress("Measuring", len(outputs)) as pb:
            for key, out in outputs:
                key_kwargs = keys | dict(zip(outputs.key_fields, key))
                list_test = test.lookup_projected(key)

                if list_test is None:
                    no_test_count += 1
                    list_test = ItemList([])

                self.measure_list(out, list_test, **key_kwargs)
                pb.update()

        if no_test_count:
            _log.warning("could not find test data for %d lists", no_test_count)

    def list_metrics(self) -> pd.DataFrame:
        """
        Get the per-list metric results as a DataFrame.

        Returns:
            DataFrame with one row per list and one column per metric.
        """

        df = pd.DataFrame.from_records(self._list_records)
        if self.key_fields:
            df.set_index(self.key_fields, inplace=True, drop=True)

        return df

    def summary_metrics(self) -> dict[str, float]:
        """
        Compute summary statistics by calling each metric's summarize() method.

        .. note::

            This returns *overall* summaries — summaries are not collected
            separately for different calls to :meth:`measure_collection`.

        Returns:
            A dictionary with flattened metric results.
        """
        results = {}

        for state in self._metrics:
            agg = state.accumulator.accumulate()
            _add_values(results, state.label, agg)

        return results

    def _validate_setup(self):
        """Validate that the accumulator is properly configured."""
        seen = set()
        for m in self._metrics:
            lbl = m.label
            if lbl in seen:
                raise RuntimeError(f"duplicate metric: {lbl}")
            seen.add(lbl)


def _wrap_metric(
    m: Metric | MetricFunction | type[Metric],
    label: str | None = None,
) -> MetricState:
    """Wrap a metric with its configuration."""
    if isinstance(m, type):
        m = m()
    elif not isinstance(m, Metric):
        m = FunctionMetric(m)
    assert isinstance(m, Metric), f"invalid type for metric {m}"

    if label is None:
        wl = m.label
    else:
        wl = label

    return MetricState(m, m.create_accumulator(), wl)


def _add_values(record: dict[str, MetricVal], name: str, data: MetricResult | None):
    """
    Add metric values to a record dictionary, a specified metric name.

    Args:
        record:
            The record to receive the data.
        name:
            The metric name for the data.
        data:
            The input data values.
    """
    if data is None:
        return
    elif isinstance(data, Mapping):
        for k, v in data.items():
            record[f"{name}.{k}"] = v
    else:
        record[name] = data
