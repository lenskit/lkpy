# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from lenskit.data import ItemList
from lenskit.diagnostics import DataWarning

from ._base import Metric, MetricFunction

_log = logging.getLogger(__name__)


def to_metric_dict(label: str, value: float | dict[str, float] | None) -> dict[str, float]:
    """
    Normalize a metric result into a flat dictionary.
    """
    if value is None:
        return {}
    elif isinstance(value, (float, int, np.floating, np.integer)):
        return {label: float(value)}
    elif isinstance(value, dict):
        return {f"{label}.{k}": float(v) for k, v in value.items()}
    else:
        raise TypeError(f"{label}: unsupported metric result type {type(value)}")


class MetricAccumulator:
    """
    Handles metric registration, accumulation, and summary aggregation.
    """

    def __init__(self):
        self._metrics: list[Metric | MetricFunction] = []
        self._labels: list[str] = []
        self._defaults: dict[str, float] = {}
        self._results: list[dict[str, float]] = []
        self._intermediates: list[dict[str, Any]] = []

    def add_metric(
        self,
        metric: Metric | MetricFunction | type[Metric],
        label: str | None = None,
        default: float | None = None,
    ) -> None:
        """
        Register a metric to be accumulated.

        Args:
            metric: An instance of a Metric subclass, a callable metric function,
                    or a Metric class to instantiate.
            label: The label to use for the metric's results. If unset, obtains
                    from the metric.
            default: The default value to use in aggregates when a user does not
                    have recommendations. If unset, obtains from the metric's
                    ``default`` attribute (if specified), or 0.0.
        """
        if isinstance(metric, type):
            metric = metric()

        # determine label
        if label is None:
            if isinstance(metric, Metric):
                metric_label = metric.label
            else:
                metric_label = metric.__name__  # type: ignore
        else:
            metric_label = label

        # determine default value
        if default is None:
            default = getattr(metric, "default", 0.0)

        if not isinstance(default, (float, int, np.floating, np.integer)):
            raise TypeError(f"metric {metric} has unsupported default {default}")

        # check for duplicate labels
        if metric_label in self._labels:
            raise ValueError(f"Metric label '{metric_label}' already exists")

        self._metrics.append(metric)
        self._labels.append(metric_label)
        self._defaults[metric_label] = float(default)

    def measure_list(self, output: ItemList, test: ItemList, **keys) -> None:
        """
        Measure and accumulate metrics for a single recommendation list.

        Args:
            output: The recommendation items for a user.
            test: The ground-truth items for the same user.
            keys: Identifier for the user (e.g., user_id).
        """
        record = dict(**keys)
        intermediate = dict(**keys)

        for metric, label in zip(self._metrics, self._labels):
            try:
                if isinstance(metric, Metric):
                    result = metric.measure_list(output, test)
                    metrics = metric.extract_list_metrics(result)
                    record.update(to_metric_dict(label, metrics))
                    intermediate[label] = result
                elif callable(metric):
                    result = metric(output, test)
                    record.update(to_metric_dict(label, result))
                    intermediate[label] = result
                else:
                    _log.warning(f"Metric {metric} is not a supported metric type")
            except Exception as e:
                _log.warning(f"Error computing metric {label}: {e}")
                intermediate[label] = None

        self._results.append(record)
        self._intermediates.append(intermediate)

    def list_metrics(self, fill_missing: bool = True) -> pd.DataFrame:
        """
        Get accumulated per-list metric results.

        Args:
            fill_missing: If True, fills in missing values with each metric's
            default value when available.

        Returns:
            A DataFrame where each row corresponds to a user,
            and columns correspond to metrics.
        """
        df = pd.DataFrame(self._results)
        if fill_missing and self._defaults:
            df = df.fillna(self._defaults)
        return df

    def summary_metrics(self) -> dict[str, float]:
        """
        Compute overall summary statistics by aggregating per-list metrics.

        Returns:
            A dictionary mapping metric labels to their aggregated summary values.
        """
        summaries = {}

        for metric, label in zip(self._metrics, self._labels):
            try:
                values = [
                    r[label] for r in self._intermediates if label in r and r[label] is not None
                ]
                if not values:
                    continue

                if isinstance(metric, Metric) and hasattr(metric, "summarize"):
                    summary = metric.summarize(values)
                else:
                    summary = np.mean(values)
                summaries.update(to_metric_dict(label, summary))
            except Exception as e:
                _log.warning(f"Error computing summary for metric {label}: {e}")

        return summaries

    def to_result(self):
        """
        Convert the accumulator result into a RunAnalysisResult.

        Returns:
            A RunAnalysisResult with per-user metrics, global summary, and defaults.
        """
        from lenskit.metrics.bulk import RunAnalysisResult

        summary = self.summary_metrics()

        # fill any missing summaries with defaults
        for key, value in self._defaults.items():
            summary.setdefault(key, value)

        return RunAnalysisResult(
            self.list_metrics(fill_missing=True),
            pd.Series(summary, dtype=np.float64),
            self._defaults,
        )

    def clear(self) -> None:
        """
        Clear all accumulated results while keeping the registered metrics.
        """
        self._results.clear()
        self._intermediates.clear()

    def __len__(self) -> int:
        """
        Return the number of accumulated measurement records.
        """
        return len(self._results)
