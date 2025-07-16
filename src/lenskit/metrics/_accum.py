# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa

from lenskit.data import ItemList

from ._base import DecomposedMetric, GlobalMetric, ListMetric, Metric, MetricFunction

_log = logging.getLogger(__name__)
K1 = TypeVar("K1", bound=tuple)
K2 = TypeVar("K2", bound=tuple)


@dataclass(frozen=True)
class MetricWrapper:
    """
    Internal class for storing metrics with their configuration.

    Stability:
        Internal
    """

    metric: Metric | MetricFunction
    label: str
    default: float | None = None

    @property
    def is_listwise(self) -> bool:
        "Check if this metric is listwise."
        return isinstance(self.metric, (ListMetric, Callable))

    @property
    def is_global(self) -> bool:
        "Check if this metric is global."
        return isinstance(self.metric, GlobalMetric)

    @property
    def is_decomposed(self) -> bool:
        "Check if this metric is decomposed."
        return isinstance(self.metric, DecomposedMetric)

    def measure_list(self, list: ItemList, test: ItemList) -> float | dict[str, float]:
        """Measure a single list and return metric result(s)."""
        if isinstance(self.metric, ListMetric):
            return self.metric.measure_list(list, test)
        elif isinstance(self.metric, Callable):
            return self.metric(list, test)
        else:
            raise TypeError(f"metric {self.metric} does not support list measurement")

    def compute_list_data(self, list: ItemList, test: ItemList) -> Any:
        """Compute intermediate data for a single list."""
        if isinstance(self.metric, (DecomposedMetric, Metric)):
            if hasattr(self.metric, "measure_list") and callable(
                getattr(self.metric, "measure_list")
            ):
                return self.metric.measure_list(list, test)
            else:
                raise TypeError(f"metric {self.metric} does not support list data computation")
        else:
            raise TypeError(f"metric {self.metric} does not support list data computation")

    def extract_list_metrics(self, data: Any) -> float | dict[str, float] | None:
        """Extract per-list metrics from intermediate data."""
        if isinstance(self.metric, (DecomposedMetric, Metric)):
            # Use new unified interface if available, fall back to old
            if hasattr(self.metric, "extract_list_metrics"):
                return self.metric.extract_list_metrics(data)
            else:
                return data if isinstance(data, (float, dict)) else None
        return None

    def summarize(self, values: list[Any] | pa.Array | pa.ChunkedArray) -> float | dict[str, float]:
        """Aggregate intermediate values into summary statistics."""
        if isinstance(self.metric, (DecomposedMetric, Metric)):
            if hasattr(self.metric, "summarize"):
                return self.metric.summarize(values)
            else:
                return self._default_summarize(values)
        else:
            return self._default_summarize(values)

    def _default_summarize(self, values) -> dict[str, float]:
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            values = values.to_pylist()
        numeric_values = [v for v in values if v is not None]
        if not numeric_values:
            return {"mean": 0.0}
        return {"mean": float(np.mean(numeric_values))}


class MetricAccumulator:
    """
    Accumulates metric measurements over multiple recommendation lists.

    This class seperates metric accumulation from the evaluation loop.

    Stability:
        Caller
    """

    def __init__(self):
        self.metrics: list[MetricWrapper] = []
        self._list_data: dict[str, list[Any]] = {}
        self._list_metrics: dict[str, list[float | dict[str, float] | None]] = {}
        self._list_keys: list[tuple] = []
        self._key_fields: list[str] = []

    def add_metric(
        self,
        metric: Metric | MetricFunction | type[Metric],
        label: str | None = None,
        default: float | None = None,
    ):
        """
        Add a metric to this accumulator.

        Args:
            metric:
                The metric to add.
            label:
                The label to use for the metric's results. If unset, obtains
                from the metric.
            default:
                The default value to use when a user does not have
                recommendations. If unset, obtains from the metric's ``default``
                attribute (if specified), or 0.0.
        """
        wrapper = self._wrap_metric(metric, label, default)
        self.metrics.append(wrapper)
        self._list_data[wrapper.label] = []
        self._list_metrics[wrapper.label] = []

    def measure_list(self, output: ItemList, test: ItemList, **keys: Any):
        """
        Measure a single list and accumulate the results.

        Args:
            output: The recommendation list to measure.
            test: The ground truth test data.
            **keys: Identifying keys for this list (e.g., user_id).
        """
        if not self._key_fields:
            self._key_fields = list(keys.keys())

        key_tuple = tuple(keys.get(k) for k in self._key_fields)
        self._list_keys.append(key_tuple)

        for wrapper in self.metrics:
            try:
                if wrapper.is_decomposed:
                    data = wrapper.compute_list_data(output, test)
                    self._list_data[wrapper.label].append(data)
                    list_metric = wrapper.extract_list_metrics(data)
                    self._list_metrics[wrapper.label].append(list_metric)
                else:
                    metric_value = wrapper.measure_list(output, test)
                    self._list_data[wrapper.label].append(metric_value)
                    self._list_metrics[wrapper.label].append(metric_value)
            except Exception as e:
                _log.warning(f"Error computing metric {wrapper.label}: {e}")
                self._list_data[wrapper.label].append(None)
                self._list_metrics[wrapper.label].append(None)

    def list_metrics(self, fill_missing: bool = True) -> pd.DataFrame:
        """
        Get the per-list metric results as a DataFrame.

        Args:
            fill_missing: If True, fill missing values with metric defaults.

        Returns:
            DataFrame with one row per list and one column per metric.
        """
        if not self._list_keys:
            return pd.DataFrame()

        if self._key_fields:
            index = pd.MultiIndex.from_tuples(self._list_keys, names=self._key_fields)
        else:
            index = pd.Index(self._list_keys)

        data = {}
        for wrapper in self.metrics:
            metric_results = self._list_metrics[wrapper.label]

            if metric_results and isinstance(metric_results[0], dict):
                for result in metric_results:
                    if isinstance(result, dict):
                        for sub_key, sub_value in result.items():
                            col_name = f"{wrapper.label}.{sub_key}"
                            if col_name not in data:
                                data[col_name] = []
                            data[col_name].append(sub_value)
            else:
                data[wrapper.label] = metric_results

        df = pd.DataFrame(data, index=index)

        if fill_missing:
            defaults = {
                wrapper.label: wrapper.default
                for wrapper in self.metrics
                if wrapper.default is not None
            }
            df = df.fillna(defaults)

        return df

    def summary_metrics(self) -> pd.DataFrame:
        summaries = {}

        for wrapper in self.metrics:
            data = self._list_data[wrapper.label]
            clean_data = [x for x in data if x is not None]

            if clean_data:
                try:
                    summary = wrapper.summarize(clean_data)
                    if isinstance(summary, dict):
                        summaries[wrapper.label] = summary
                    else:
                        summaries[wrapper.label] = {"mean": summary}
                except Exception as e:
                    _log.warning(f"Error summarizing metric {wrapper.label}: {e}")
                    summaries[wrapper.label] = {"mean": 0.0}
            else:
                summaries[wrapper.label] = {"mean": 0.0}

        if summaries:
            df = pd.DataFrame(summaries).T
            df.index.name = "metric"
            return df
        else:
            return pd.DataFrame()

    def _wrap_metric(
        self,
        m: Metric | MetricFunction | type[Metric],
        label: str | None = None,
        default: float | None = None,
    ) -> MetricWrapper:
        """Wrap a metric with its configuration."""
        if isinstance(m, type):
            m = m()

        if label is None:
            if isinstance(m, Metric):
                wl = m.label
            else:
                wl = m.__name__  # type: ignore
        else:
            wl = label

        if default is None:
            if isinstance(m, ListMetric):
                default = m.default
            else:
                default = 0.0

            if default is not None and not isinstance(
                default, (float, int, np.floating, np.integer)
            ):
                raise TypeError(f"metric {m} has unsupported default {default}")

        return MetricWrapper(m, wl, default)  # type: ignore

    def _validate_setup(self):
        """Validate that the accumulator is properly configured."""
        seen = set()
        for m in self.metrics:
            lbl = m.label
            if lbl in seen:
                raise RuntimeError(f"duplicate metric: {lbl}")
            seen.add(lbl)


def to_result(acc: MetricAccumulator):
    """
    Convert a MetricAccumulator into a RunAnalysisResult.
    """
    from .bulk import RunAnalysisResult

    list_results = acc.list_metrics(fill_missing=False)
    global_results = acc.summary_metrics()["mean"]

    defaults = {wrapper.label: wrapper.default for wrapper in acc.metrics}

    return RunAnalysisResult(list_results, global_results, defaults)  # type: ignore
