from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import pandas as pd
from progress_api import make_progress

from lenskit.data import ItemListCollection

from ._base import Metric, MetricFunction

_log = logging.getLogger(__name__)
K1 = TypeVar("K1", bound=tuple)
K2 = TypeVar("K2", bound=tuple)


@dataclass(frozen=True)
class MetricWrapper:
    """
    Internal class for storing metrics.
    """

    metric: Metric | MetricFunction
    label: str
    mean_label: str
    default: float | None = None


class RunAnalysisResult:
    """
    Results of a bulk metric computation.
    """

    _list_scores: pd.DataFrame
    _defaults: dict[str, float]

    def __init__(self, lscores: pd.DataFrame, defaults: dict[str, float]):
        self._list_scores = lscores
        self._defaults = defaults

    def list_scores(self, *, fill_missing=True) -> pd.DataFrame:
        """
        Get the per-list scores of the results.  This is a data frame with one
        row per list (with the list / user ID in the index), and one metric per
        column.

        Args:
            fill_missing:
                If ``True`` (the default), fills in missing values with each
                metric's default value when available.  Pass ``False`` if you
                want to do analyses that need to treat missing values
                differently.
        """
        return self._list_scores.fillna(self._defaults)

    def summary(self) -> pd.DataFrame:
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
        """
        scores = self.list_scores(fill_missing=True)
        return scores.agg(["mean", "median", "std"]).T


def _wrap_metric(
    m: Metric | MetricFunction | type[Metric],
    label: str | None = None,
    mean_label: str | None = None,
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

    if mean_label is None:
        if label is not None:
            wml = label
        elif isinstance(m, Metric):
            wml = m.mean_label
        else:
            wml = wl

    if default is None:
        if isinstance(m, Metric):
            default = m.default
        else:
            default = 0.0

        if default is not None and not isinstance(default, (float, int, np.floating, np.integer)):
            raise TypeError(f"metric {m} has unsupported default {default}")

    return MetricWrapper(m, wl, wml, default)  # type: ignore


class RunAnalysis:
    """
    Compute metrics over a collection of item lists composing a run.

    Args:
        metrics:
            A list of metrics; you can also add them with :meth:`add_metric`,
            which provides more flexibility.
    """

    metrics: list[MetricWrapper]
    "The list of metrics to compute."

    def __init__(self, *metrics: Metric):
        self.metrics = [_wrap_metric(m) for m in metrics]

    def add_metric(
        self,
        metric: Metric | MetricFunction | type[Metric],
        label: str | None = None,
        mean_label: str | None = None,
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
            mean_label:
                The label to use for the overall aggregate (mean) of the
                metric's results.  If unset, obtains from the metric.
            default:
                The default value to use in aggregates when a user does not have
                recommendations. If unset, obtains from the metric's ``default``
                attribute (if specified), or 0.0.
        """
        self.metrics.append(_wrap_metric(metric, label, mean_label, default))

    def compute(
        self, outputs: ItemListCollection[K1], test: ItemListCollection[K2]
    ) -> RunAnalysisResult:
        index = pd.MultiIndex.from_tuples(outputs.keys())
        results = pd.DataFrame({m.label: np.nan for m in self.metrics}, index=index)

        n = len(outputs)
        _log.info("computing metrics for %d output lists", n)
        with make_progress(_log, "lists", n) as pb:
            for i, (key, out) in enumerate(outputs):
                list_test = test.lookup_projected(key)
                if out is None:
                    pass
                elif list_test is None:
                    _log.warning("list %s: no test items", key)
                else:
                    results.iloc[i] = [m.metric(out, list_test) for m in self.metrics]
                pb.update()

        return RunAnalysisResult(
            results, {m.label: m.default for m in self.metrics if m.default is not None}
        )
