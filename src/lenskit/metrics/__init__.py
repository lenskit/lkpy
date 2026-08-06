# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Metrics for evaluating recommender outputs.
"""

import warnings
from collections.abc import Callable

from lenskit.data import ItemList

from ._base import ListMetric, Metric, MetricFunction, MetricResult, MetricVal
from ._collect import MeasurementCollector, RunMetrics
from ._quick import quick_measure_model
from .basic import ListLength, TestItemCount, UniqueItemCount
from .bulk import RunAnalysis, RunAnalysisResult
from .predict import MAE, RMSE
from .ranking import (
    DCG,
    ILS,
    IPSRBP,
    NDCG,
    RBP,
    Entropy,
    ExposureGini,
    FieldPropensity,
    GeometricRankWeight,
    Hit,
    ListGini,
    LogRankWeight,
    MeanPopRank,
    PopularityPropensity,
    Precision,
    PropensityModel,
    RankBiasedEntropy,
    RankingMetricBase,
    RankWeight,
    Recall,
    RecipRank,
    UniformPropensity,
    estimate_power_law_gamma,
)
from .reranking import least_item_promoted, rank_biased_overlap

__all__ = [
    "DCG",
    "ILS",
    "IPSRBP",
    "MAE",
    "NDCG",
    "RBP",
    "RMSE",
    "Entropy",
    "ExposureGini",
    "FieldPropensity",
    "GeometricRankWeight",
    "Hit",
    "ListGini",
    "ListLength",
    "ListMetric",
    "LogRankWeight",
    "MeanPopRank",
    "MeasurementCollector",
    "Metric",
    "MetricFunction",
    "MetricResult",
    "MetricVal",
    "PopularityPropensity",
    "Precision",
    "PropensityModel",
    "RankBiasedEntropy",
    "RankWeight",
    "RankingMetricBase",
    "Recall",
    "RecipRank",
    "RunAnalysis",
    "RunAnalysisResult",
    "RunMetrics",
    "TestItemCount",
    "UniformPropensity",
    "UniqueItemCount",
    "estimate_power_law_gamma",
    "least_item_promoted",
    "quick_measure_model",
    "rank_biased_overlap",
]


def call_metric[**P](
    metric: Metric | MetricFunction | Callable[P, Metric],
    outs: ItemList,
    test: ItemList | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> MetricResult | None:
    """
    Deprecated alias for :func:`measure_list`.

    .. deprecated:: 2026.1

        Use :func:`measure_list`.
    """
    warnings.warn("call_metric is deprecated, use measure_list instead", DeprecationWarning)
    return measure_list(metric, outs, test, *args, **kwargs)


def measure_list[**P](
    metric: Metric | MetricFunction | Callable[P, Metric],
    outs: ItemList,
    test: ItemList | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> MetricResult | None:
    """
    Call a metric to measure a list, instantiating it if necessary.  This
    intended to be a quick convenience when you just need to call a metric with
    its default settings and don't want to mess around with object/class
    distinctions.  You usually don't actually want to use it.


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

    if isinstance(metric, Metric):
        x = metric.measure_list(outs, test)  # type: ignore
        return metric.extract_list_metrics(x)
    elif isinstance(metric, Callable):
        return metric(outs, test)  # type: ignore
    else:  # pragma: nocover
        raise TypeError(f"invalid metric {metric}")
