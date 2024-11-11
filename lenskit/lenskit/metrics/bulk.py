from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, TypeAlias

import numpy as np
import pandas as pd
from progress_api import make_progress

from lenskit.data import EntityId, ItemList
from lenskit.data.bulk import count_item_lists, dict_from_df, iter_item_lists

from ._base import LabeledMetric, Metric

_log = logging.getLogger(__name__)
ILSet: TypeAlias = Mapping[EntityId, ItemList | None] | pd.DataFrame


@dataclass(frozen=True)
class MetricWrapper:
    """
    Internal class for storing metrics.
    """

    metric: Metric
    label: str
    mean_label: str
    default: float


class RunAnalysisResult:
    """
    Results of a bulk metric computation.
    """

    list_scores: pd.DataFrame
    """
    The metric scores for each list.  The row indices are list (user) IDs, and
    there is one column for each metric.
    """

    _defaults: dict[str, float]

    def __init__(self, lscores: pd.DataFrame, defaults: dict[str, float]):
        self.list_scores = lscores
        self._defaults = defaults

    def summary(self) -> pd.DataFrame:
        """
        Sumamry statistics for the per-list metrics.  Each metric is on its own row,
        with columns reporting the following:

        ``mean``:
            The mean metric value.
        ``median``:
            The median metric value.
        ``sd``:
            The standard deviation of the metric.

        Additional columns are added based on other options.
        """
        df = self.list_scores.fillna(self._defaults)
        return pd.DataFrame({"mean": df.mean()})


def _wrap_metric(
    m: Metric, label: str | None = None, mean_label: str | None = None, default: float | None = None
) -> MetricWrapper:
    if label is None:
        if isinstance(m, LabeledMetric):
            wl = m.label
        else:
            wl = m.__name__  # type: ignore
    else:
        wl = label

    if mean_label is None:
        if label is not None:
            wml = label
        elif isinstance(m, LabeledMetric):
            wml = m.mean_label
        else:
            wml = wl

    if default is None:
        default = getattr(m, "default", None)
        if default is None:
            default = 0.0
        elif not isinstance(default, (float, int, np.floating, np.integer)):
            raise TypeError(f"metric {m} has unsupported default {default}")

    return MetricWrapper(m, wl, wml, default)


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
        metric: Metric,
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

    def compute(self, outputs: ILSet, test: ILSet) -> RunAnalysisResult:
        if isinstance(test, pd.DataFrame):
            test = dict_from_df(test)

        columns = [m.label for m in self.metrics]
        user_ids = []
        rows = []

        n = count_item_lists(outputs)
        _log.info("computing metrics for %d output lists", n)
        with make_progress(_log, "lists", n) as pb:
            for uid, out in iter_item_lists(outputs):
                list_test = test[uid]
                if out is None:
                    rows.append([None] for _i in range(len(columns)))
                elif list_test is None:
                    _log.warning("list %s: no test items", uid)
                else:
                    row = [m.metric(out, list_test) for m in self.metrics]
                    user_ids.append(uid)
                    rows.append(row)
                pb.update()

        df = pd.DataFrame.from_records(rows, index=user_ids, columns=columns)
        return RunAnalysisResult(df, {m.label: m.default for m in self.metrics})
