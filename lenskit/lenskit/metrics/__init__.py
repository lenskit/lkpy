# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Metrics for evaluating recommender outputs.
"""

from typing import Callable, ParamSpec

from lenskit.data import ItemList

from ._base import ListMetric, Metric, MetricFunction
from ._quick import quick_measure_model
from .bulk import RunAnalysis, RunAnalysisResult
from .predict import MAE, RMSE
from .ranking import NDCG, RBP, Precision, RankingMetricBase, Recall, RecipRank

__all__ = [
    "Metric",
    "MetricFunction",
    "RankingMetricBase",
    "RunAnalysis",
    "RunAnalysisResult",
    "RMSE",
    "MAE",
    "NDCG",
    "RBP",
    "Precision",
    "Recall",
    "RecipRank",
    "quick_measure_model",
]

P = ParamSpec("P")


def call_metric(
    metric: ListMetric | MetricFunction | Callable[P, ListMetric],
    outs: ItemList,
    test: ItemList | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> float:
    """
    Call a metric, instantiating it if necessary.  This intended to be a quick
    convenience when you just need to call a metric with its default settings
    and don't want to mess around with object/class distinctions.  You usually
    don't actually want to use it.

    Supports both the base :class:`Metric` protocol and the extensions in
    :class:`PredictMetric`.

    Args:
        metric:
            The metric to call.  Note that the type of ``Callable`` is slightly
            too loose, it actually needs to be a class that subclasses
            :class:`Metric`.
        outs:
            The output to measure.
        test:
            The test data to measure.
    """

    if isinstance(metric, type):
        metric = metric(*args, **kwargs)

    if isinstance(metric, ListMetric):
        return metric.measure_list(outs, test)  # type: ignore
    elif isinstance(metric, Callable):
        return metric(outs, test)  # type: ignore
    else:  # pragma: nocover
        raise TypeError(f"invalid metric {metric}")
