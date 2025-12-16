# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Metrics for evaluating recommender outputs.
"""

from typing import Callable, ParamSpec

from lenskit.data import ItemList

from ._base import GlobalMetric, ListMetric, Metric, MetricFunction
from ._collect import MeasurementCollector
from ._quick import quick_measure_model
from .basic import ListLength, TestItemCount
from .bulk import RunAnalysis, RunAnalysisResult
from .predict import MAE, RMSE
from .ranking import (
    DCG,
    ILS,
    NDCG,
    RBP,
    Entropy,
    ExposureGini,
    GeometricRankWeight,
    Hit,
    ListGini,
    LogRankWeight,
    MeanPopRank,
    Precision,
    RankBiasedEntropy,
    RankingMetricBase,
    RankWeight,
    Recall,
    RecipRank,
)
from .reranking import least_item_promoted, rank_biased_overlap

__all__ = [
    "Metric",
    "MetricFunction",
    "MeasurementCollector",
    "ListMetric",
    "GlobalMetric",
    "RankingMetricBase",
    "RunAnalysis",
    "RunAnalysisResult",
    "ListLength",
    "TestItemCount",
    "RankWeight",
    "GeometricRankWeight",
    "LogRankWeight",
    "RMSE",
    "MAE",
    "NDCG",
    "DCG",
    "RBP",
    "Hit",
    "Precision",
    "Recall",
    "RecipRank",
    "MeanPopRank",
    "ListGini",
    "ExposureGini",
    "quick_measure_model",
    "least_item_promoted",
    "rank_biased_overlap",
    "ILS",
    "Entropy",
    "RankBiasedEntropy",
]

P = ParamSpec("P")
MetricAccumulator = MeasurementCollector
"""
Deprecated alias for :class:`MeasurementCollector`.

.. deprecated:: 2025.5
    Use the new name.
"""


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
