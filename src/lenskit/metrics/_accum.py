# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from lenskit.data import ItemList

from ._base import Metric

_log = logging.getLogger(__name__)


class MetricAccumulator:
    """
    Handles metric registration, accumulation of metric measurements
    per recommendation list, and aggregation of summary statistics.
    """

    def __init__(self):
        self._metrics: list[Metric] = []
        self._results: list[dict[str, float]] = []
        self._intermediates: list[dict[str, Any]] = []

    def add_metric(self, metric: Metric) -> None:
        """
        Register a metric to be accumulated.

        Args:
            metric: An instance of a Metric subclass.
        """
        self._metrics.append(metric)

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

        for metric in self._metrics:
            result = metric.measure_list(output, test)
            extracted = metric.extract_list_metrics(result)
            record.update(normalize_metrics(metric.label, extracted))
            intermediate[metric.label] = result

        self._results.append(record)
        self._intermediates.append(intermediate)

    def list_metrics(self) -> pd.DataFrame:
        """
        Get accumulated per-list metric results.

        Returns:
            A DataFrame where each row corresponds to a user,
            and columns correspond to metrics.
        """
        return pd.DataFrame(self._results)

    def summary_metrics(self) -> dict[str, float]:
        """
        Compute overall summary statistics by aggregating per-list metrics.

        Returns:
            A dictionary mapping metric labels to their aggregated summary values.
        """
        summaries = {}
        for metric in self._metrics:
            label = metric.label
            values = [r[label] for r in self._intermediates if label in r]
            if not values:
                _log.warning("No intermediate values for metric '%s'; skipping summary.", label)
                continue
            result = metric.summarize(values)
            summaries.update(normalize_metrics(label, result))
        return summaries

    def to_result(self, defaults: dict[str, float]):
        from lenskit.metrics.bulk import RunAnalysisResult

        """
        Convert the accumulator result into a RunAnalysisResult.

        Args:
            defaults: A dictionary of default values to return if a metric is missing.

        Returns:
            A RunAnalysisResult with per-user metrics, global summary, and defaults.
        """
        summary = self.summary_metrics()
        for key, value in defaults.items():
            summary.setdefault(key, value)

        return RunAnalysisResult(
            self.list_metrics(),
            pd.Series(summary),
            defaults,
        )


def normalize_metrics(label: str, value: float | dict[str, float] | None) -> dict[str, float]:
    """
    Normalize a metric result into a flat dictionary.
    """
    if value is None:
        return {}
    elif isinstance(value, (float, int)):
        return {label: float(value)}
    elif isinstance(value, dict):
        return {f"{label}.{k}": float(v) for k, v in value.items()}
    else:
        raise TypeError(f"{label}: unsupported metric result type {type(value)}")
