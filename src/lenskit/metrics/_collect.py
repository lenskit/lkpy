# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from lenskit.data import ItemList, ItemListCollection
from lenskit.data.accum import Accumulator

from ._base import (
    FunctionMetric,
    GlobalMetric,
    Metric,
    MetricFunction,
    MetricResult,
    MetricVal,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
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

    def measure_run(self, run: ItemListCollection, test: ItemListCollection) -> float:
        """Only global metrics support run-level measurement."""
        if isinstance(self.metric, GlobalMetric):
            return self.metric.measure_run(run, test)
        else:
            raise TypeError(f"metric {self.metric} does not support global measurement")


class MeasurementCollector:
    """
    Collect metric measurements over multiple recommendation lists.

    This class separates metric collection and aggregation from the main
    evaluation loop.

    .. versionchanged:: 2026.1

        ``metrics`` is no longer publicly exposed as a list of wrappers.

    Stability:
        Caller
    """

    _metrics: list[MetricState]
    _list_records: list[dict[str, float | int | object]]
    _key_fields: list[str]

    def __init__(self):
        self._metrics = []
        self._list_records = []
        self._key_fields = []

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
        if not self._key_fields:
            self._key_fields = list(keys.keys())

        rec = dict(keys)
        for state in self._metrics:
            intermediate = state.metric.measure_list(output, test)
            state.accumulator.add(intermediate)
            lv = state.metric.extract_list_metrics(intermediate)
            _add_values(rec, state.label, lv)

        self._list_records.append(rec)

    def list_metrics(self, fill_missing: bool = True) -> pd.DataFrame:
        """
        Get the per-list metric results as a DataFrame.

        Args:
            fill_missing: If True, fill missing values with metric defaults.

        Returns:
            DataFrame with one row per list and one column per metric.
        """

        df = pd.DataFrame.from_records(self._list_records)
        if self._key_fields:
            df.set_index(self._key_fields, inplace=True, drop=True)

        return df

    def summary_metrics(self) -> dict[str, float]:
        """
        Compute summary statistics by calling each metric's summarize() method.

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
        if isinstance(m, Metric):
            wl = m.label
        elif isinstance(m, type):
            wl = m.__name__  # type: ignore
        else:
            wl = type(m).__name__
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
