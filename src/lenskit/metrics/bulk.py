# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

from lenskit.data import ItemList, ItemListCollection
from lenskit.diagnostics import DataWarning
from lenskit.logging import item_progress

from ._base import DecomposedMetric, GlobalMetric, ListMetric, Metric, MetricFunction

_log = logging.getLogger(__name__)
K1 = TypeVar("K1", bound=tuple)
K2 = TypeVar("K2", bound=tuple)


@dataclass(frozen=True)
class MetricWrapper:
    """
    Internal class for storing metrics.

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

    def measure_list(self, list: ItemList, test: ItemList) -> float:
        if isinstance(self.metric, ListMetric):
            return self.metric.measure_list(list, test)
        elif isinstance(self.metric, Callable):
            return self.metric(list, test)
        else:
            raise TypeError(f"metric {self.metric} does not support list measurement")

    def measure_run(self, run: ItemListCollection, test: ItemListCollection) -> float:
        if isinstance(self.metric, GlobalMetric):
            return self.metric.measure_run(run, test)
        else:
            raise TypeError(f"metric {self.metric} does not support global measurement")


class RunAnalysisResult:
    """
    Results of a bulk metric computation.

    Stability:
        Caller
    """

    _list_metrics: pd.DataFrame
    _global_metrics: pd.Series
    _defaults: dict[str, float]

    def __init__(self, lmvs: pd.DataFrame, gmvs: pd.Series, defaults: dict[str, float]):
        self._list_metrics = lmvs
        self._global_metrics = gmvs
        self._defaults = defaults

    def global_metrics(self) -> pd.Series:
        """
        Get the global metric scores.  This is only the results of
        global metrics; it does not include aggregates of per-list metrics.  For
        aggregates of per-list metrics, call :meth:`list_summary`.
        """
        return self._global_metrics

    def list_metrics(self, fill_missing=True) -> pd.DataFrame:
        """
        Get the per-list scores of the results.  This is a data frame with one
        row per list (with the list key on the inded), and one metric per
        column.

        Args:
            fill_missing:
                If ``True`` (the default), fills in missing values with each
                metric's default value when available.  Pass ``False`` if you
                want to do analyses that need to treat missing values
                differently.
        """
        return self._list_metrics.fillna(self._defaults)

    def list_summary(self, *keys: str) -> pd.DataFrame:
        """
        Sumamry statistics for the per-list metrics.  Each metric is on its own
        row, with columns reporting the following:

        ``mean``:
            The mean metric value.
        ``median``:
            The median metric value.
        ``std``:
            The (sample) standard deviation of the metric.

        Additional columns are added based on other options.  Missing metric
        values are filled with their defaults before computing statistics.

        Args:
            keys:
                Identifiers for different conditions that should be reported
                separately (grouping keys for the final result).
        """
        scores = self.list_metrics(fill_missing=True)
        if keys:
            df = scores.groupby(list(keys)).agg(["mean", "median", "std"]).stack(level=0)
            assert isinstance(df, pd.DataFrame)
        else:
            df = scores.agg(["mean", "median", "std"]).T
            df.index.name = "metric"
        return df

    def merge_from(self, other: RunAnalysisResult):
        """
        Merge another set of analysis results into this one.
        """
        for c in self._list_metrics.columns:
            if c in other._list_metrics.columns:
                warnings.warn(f"list metric {c} appears in both merged results", DataWarning)

        for c in self._global_metrics.index:
            if c in other._global_metrics.index:
                warnings.warn(f"global metric {c} appears in both merged results", DataWarning)

        self._list_metrics = self._list_metrics.join(other._list_metrics, how="outer")
        self._global_metrics = pd.concat([self._global_metrics, other._global_metrics])


def _wrap_metric(
    m: Metric | MetricFunction | type[Metric],
    label: str | None = None,
    default: float | None = None,
) -> MetricWrapper:
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

        if default is not None and not isinstance(default, (float, int, np.floating, np.integer)):
            raise TypeError(f"metric {m} has unsupported default {default}")

    return MetricWrapper(m, wl, default)  # type: ignore


class RunAnalysis:
    """
    Compute metrics over a collection of item lists composing a run.

    Args:
        metrics:
            A list of metrics; you can also add them with :meth:`add_metric`,
            which provides more flexibility.

    Stability:
        Caller
    """

    metrics: list[MetricWrapper]
    "The list of metrics to compute."

    def __init__(self, *metrics: Metric):
        self.metrics = [_wrap_metric(m) for m in metrics]

    def add_metric(
        self,
        metric: Metric | MetricFunction | type[Metric],
        label: str | None = None,
        default: float | None = None,
    ):
        """
        Add a metric to this metric set.

        Args:
            metric:
                The metric to add to the set.
            label:
                The label to use for the metric's results.  If unset, obtains
                from the metric.
            default:
                The default value to use in aggregates when a user does not have
                recommendations. If unset, obtains from the metric's ``default``
                attribute (if specified), or 0.0.
        """
        self.metrics.append(_wrap_metric(metric, label, default))

    def compute(
        self, outputs: ItemListCollection[K1], test: ItemListCollection[K2]
    ) -> RunAnalysisResult:
        """
        Deprecated alias for :meth:`measure`.

        .. deprecated:: 2025.1.1
            Use :meth:`measure` instead.
        """
        return self.measure(outputs, test)

    def measure(
        self, outputs: ItemListCollection[K1], test: ItemListCollection[K2]
    ) -> RunAnalysisResult:
        """
        Measure a set of outputs against a set of dest data.
        """
        self._validate_setup()
        index = pd.MultiIndex.from_tuples(outputs.keys())
        index.names = list(outputs.key_fields)

        lms = [m for m in self.metrics if m.is_listwise or m.is_decomposed]
        gms = [m for m in self.metrics if m.is_global]
        list_results = pd.DataFrame({m.label: np.nan for m in lms}, index=index)
        list_intermediates = {m.label: [] for m in self.metrics if m.is_decomposed}

        n = len(outputs)
        _log.info("computing %d listwise metrics for %d output lists", len(lms), n)
        with item_progress("Measuring", n) as pb:
            for i, (key, out) in enumerate(outputs):
                list_test = test.lookup_projected(key)
                if out is None:
                    pass
                elif list_test is None:
                    _log.warning("list %s: no test items", key)
                else:
                    l_row: list[float | None] = []
                    for m in lms:
                        mv = None
                        if m.is_decomposed:
                            assert isinstance(m.metric, DecomposedMetric)
                            val = m.metric.compute_list_data(out, list_test)
                            list_intermediates[m.label].append(val)
                            mv = m.metric.extract_list_metric(val)
                        if mv is None and m.is_listwise:
                            mv = m.measure_list(out, list_test)
                        l_row.append(mv)
                    list_results.iloc[i] = l_row  # type: ignore
                pb.update()

        _log.info("computing %d global metrics for %d output lists", len(gms), n)
        global_results = pd.Series(
            {
                m.label: (
                    m.metric.global_aggregate(list_intermediates[m.label])
                    if m.is_decomposed
                    else m.measure_run(outputs, test)
                )
                for m in self.metrics
                if m.is_global or m.is_decomposed
            },
            dtype=np.float64,
        )

        return RunAnalysisResult(
            list_results,
            global_results,
            {m.label: m.default for m in self.metrics if m.default is not None},
        )

    def _validate_setup(self):
        seen = set()
        for m in self.metrics:
            lbl = m.label
            if lbl in seen:
                raise RuntimeError(f"duplicate metric: {lbl}")
            seen.add(lbl)
