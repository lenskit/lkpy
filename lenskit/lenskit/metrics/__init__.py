# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Metrics for evaluating recommender outputs.
"""

from typing import Callable, ParamSpec, cast, overload

import pandas as pd

from lenskit.data import ItemList

from ._base import Metric, MetricFunction
from .predict import PredictMetric

__all__ = ["Metric", "MetricFunction"]

P = ParamSpec("P")


@overload
def call_metric(
    metric: Metric | MetricFunction | Callable[P, Metric],
    outs: ItemList,
    test: ItemList,
    *args: P.args,
    **kwargs: P.kwargs,
) -> float: ...
@overload
def call_metric(
    metric: PredictMetric | Callable[P, PredictMetric],
    outs: ItemList | pd.DataFrame,
    test: ItemList | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> float: ...
def call_metric(
    metric: Metric | MetricFunction | Callable[P, Metric],
    outs: ItemList | pd.DataFrame,
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

    metric = cast(Metric | MetricFunction, metric)

    return metric(outs, test)  # type: ignore
