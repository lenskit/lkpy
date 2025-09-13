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

from lenskit.data import ItemList, ItemListCollection

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

    def measure_list(self, list: ItemList, test: ItemList) -> Any:
        """Get intermediate measurement data from the metric."""
        if isinstance(self.metric, Callable):
            return self.metric(list, test)
        return self.metric.measure_list(list, test)

    def extract_list_metrics(self, intermediate_data: Any) -> float | dict[str, float] | None:
        """Extract per-list metrics from intermediate data."""
        result = self.metric.extract_list_metrics(intermediate_data)
        return intermediate_data if result is None else result

    def summarize(self, values: list[Any] | pa.Array | pa.ChunkedArray) -> dict[str, float | None]:
        """Aggregate intermediate values into summary statistics."""
        result = self.metric.summarize(values)
        if result is None:
            return self._default_summarize(values)
        if isinstance(result, dict):
            return result
        return {"mean": float(result), "median": None, "std": None}

    def _default_summarize(self, values) -> dict[str, float | None]:
        """A fallback for metrics that do not implement their own summarize method."""
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            values = values.to_pylist()
        numeric_values = [v for v in values if v is not None]

        if not numeric_values:
            return {"mean": None, "median": None, "std": None}

        arr = np.array(numeric_values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    def measure_run(self, run: ItemListCollection, test: ItemListCollection) -> float:
        """Only global metrics support run-level measurement."""
        if isinstance(self.metric, GlobalMetric):
            return self.metric.measure_run(run, test)
        else:
            raise TypeError(f"metric {self.metric} does not support global measurement")


class MetricAccumulator:
    """
    Accumulates metric measurements over multiple recommendation lists.

    This class separates metric accumulation from the evaluation loop.

    Stability:
        Caller
    """

    def __init__(self):
        self.metrics: list[MetricWrapper] = []
        self._records: list[dict[str, Any]] = []
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
        for wrapper in self.metrics:
            if wrapper.is_global:
                continue

            intermediate_data = wrapper.measure_list(output, test)

            rec[f"_intermediate_{wrapper.label}"] = intermediate_data

        self._records.append(rec)

    def list_metrics(self, fill_missing: bool = True) -> pd.DataFrame:
        """
        Get the per-list metric results as a DataFrame.

        Args:
            fill_missing: If True, fill missing values with metric defaults.

        Returns:
            DataFrame with one row per list and one column per metric.
        """
        if not self._records:
            return pd.DataFrame()

        extracted_records = []

        for record in self._records:
            new_record = {k: v for k, v in record.items() if not k.startswith("_intermediate_")}

            for wrapper in self.metrics:
                if wrapper.is_global:
                    continue

                intermediate_key = f"_intermediate_{wrapper.label}"
                if intermediate_key in record:
                    intermediate_data = record[intermediate_key]

                    extracted_val = wrapper.extract_list_metrics(intermediate_data)

                    if extracted_val is None and wrapper.default is not None:
                        extracted_val = wrapper.default

                    if isinstance(extracted_val, dict):
                        for k, v in extracted_val.items():
                            new_record[f"{wrapper.label}.{k}"] = v
                    else:
                        new_record[wrapper.label] = extracted_val

            extracted_records.append(new_record)

        df = pd.DataFrame(extracted_records)

        if self._key_fields:
            df.set_index(self._key_fields, inplace=True, drop=True)

        if fill_missing:
            for wrapper in self.metrics:
                if wrapper.default is not None and not wrapper.is_global:
                    label = wrapper.label
                    nested_cols = [c for c in df.columns if c.startswith(f"{label}.")]
                    if nested_cols:
                        for col in nested_cols:
                            df[col] = df[col].fillna(wrapper.default)
                    elif label in df.columns:
                        df[label] = df[label].fillna(wrapper.default)

        return df

    def summary_metrics(self) -> pd.DataFrame:
        """
        Compute summary statistics by calling each metric's summarize() method.

        Returns:
            A DataFrame indexed by metric label.
        """
        summaries = {}

        for wrapper in self.metrics:
            if wrapper.is_global:
                continue

            label = wrapper.label
            intermediate_key = f"_intermediate_{label}"

            intermediate_values = []
            for record in self._records:
                if intermediate_key in record and record[intermediate_key] is not None:
                    intermediate_values.append(record[intermediate_key])

            if intermediate_values:
                summary = wrapper.summarize(intermediate_values)
            else:
                default_val = wrapper.default if wrapper.default is not None else 0.0
                summary = {"mean": default_val, "median": None, "std": None}

            summaries[label] = summary

        if not summaries:
            return pd.DataFrame(columns=["mean", "median", "std"]).rename_axis("metric")

        df = pd.DataFrame.from_dict(summaries, orient="index")
        df.index.name = "metric"
        return df

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
            elif isinstance(m, type):
                wl = m.__name__  # type: ignore
            else:
                wl = type(m).__name__
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
