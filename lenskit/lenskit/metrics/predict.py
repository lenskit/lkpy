# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Prediction accuracy metrics.  See :ref:`eval-predict-accuracy` for an overview
and instructions on using these metrics.
"""

from __future__ import annotations

from typing import Callable, Literal, TypeAlias, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lenskit.data import ItemList
from lenskit.data.bulk import group_df

from ._base import Metric

MissingDisposition: TypeAlias = Literal["error", "ignore"]
ScoreArray: TypeAlias = NDArray[np.floating] | pd.Series
PredMetric: TypeAlias = Callable[[ScoreArray, ScoreArray], float]


class PredictMetric(Metric):
    """
    Extension to the metric function interface for prediction metrics.

    In addition to the general metric interface, predict metrics can be calledin
    two additional ways:

    - Suppling a single :class:`ItemList` with both ``scores`` and a ``rating``
      field.
    - Supplying a Pandas :class:`~pd.DataFrame` with ``score`` and ``rating``
      columns. In this design, global (micro-averaged) RMSE can be computed in
      stead of the per-user RMSE computed in the default configuration.

    Args:
        missing_scores:
            The action to take when a test item has not been scored.  The
            default throws an exception, avoiding situations where non-scored
            items are silently excluded from overall statistics.
        missing_truth:
            The action to take when no test items are available for a scored
            item. The default is to also to fail; if you are scoring a superset
            of the test items for computational efficiency, set this to
            ``"ignore"``.
    """

    # predict metrics usually cannot fill in default values
    default = None
    missing_scores: MissingDisposition
    missing_truth: MissingDisposition

    def __init__(
        self,
        missing_scores: MissingDisposition = "error",
        missing_truth: MissingDisposition = "error",
    ):
        self.missing_scores = missing_scores
        self.missing_truth = missing_truth

    @overload
    def __call__(self, predictions: ItemList, truth: ItemList | None = None) -> float: ...
    @overload
    def __call__(self, predictions: pd.DataFrame) -> float: ...
    def __call__(
        self, predictions: ItemList | pd.DataFrame, truth: ItemList | None = None
    ) -> float: ...


def _score_predictions(
    metric: PredMetric,
    predictions: ItemList | pd.DataFrame,
    truth: ItemList | None = None,
    missing_scores: MissingDisposition = "error",
    missing_truth: MissingDisposition = "error",
) -> float:
    if isinstance(predictions, ItemList):
        pred_s = predictions.scores("pandas", index="ids")
        assert pred_s is not None, "item list does not have scores"
        if truth is not None:
            rate_s = truth.field("rating", "pandas", index="ids")
        else:
            rate_s = predictions.field("rating", "pandas", index="ids")
        assert rate_s is not None, "no ratings provided"
        pred_s, rate_s = pred_s.align(rate_s, join="outer")
    else:
        assert truth is None, "truth must be None when predictions is a data frame"
        if "score" in predictions.columns:
            pred_s = predictions["score"]
        elif "prediction" in predictions.columns:
            pred_s = predictions["prediction"]
        else:
            raise KeyError("predictions has neither “score” nor “prediction” columns")
        rate_s = predictions["rating"]

    pred_m = pred_s.isna()
    rate_m = rate_s.isna()

    if missing_scores == "error" and (nbad := np.sum(pred_m & ~rate_m)):
        raise ValueError(f"missing scores for {nbad} truth items")

    if missing_truth == "error" and (nbad := np.sum(rate_m & ~pred_m)):
        raise ValueError(f"missing truth for {nbad} scored items")

    keep = ~(pred_m | rate_m)

    pred_s = pred_s[keep]
    rate_s = rate_s[keep]

    return metric(pred_s, rate_s)


def _rmse(scores: ScoreArray, truth: ScoreArray) -> float:
    err = truth - scores
    return np.sqrt(np.mean(err * err)).item()


def _mae(scores: ScoreArray, truth: ScoreArray) -> float:
    err = truth - scores
    return np.mean(np.abs(err)).item()


class RMSE(PredictMetric):
    """
    Compute RMSE (root mean squared error).  This is computed as:

    .. math::
        \\sum_{r_{ui} \\in R} \\left(r_{ui} - s(i|u)\\right)^2

    This metric does not do any fallbacks; if you want to compute RMSE with
    fallback predictions (e.g. usign a bias model when a collaborative filter
    cannot predict), generate predictions with
    :class:`~lenskit.basic.FallbackScorer`.
    """

    @property
    def mean_label(self):
        return "AvgUserRMSE"

    @overload
    def __call__(self, predictions: ItemList, test: ItemList | None = None) -> float: ...
    @overload
    def __call__(self, predictions: pd.DataFrame) -> float: ...
    def __call__(self, predictions: ItemList | pd.DataFrame, test: ItemList | None = None) -> float:
        return _score_predictions(_rmse, predictions, test, self.missing_scores, self.missing_truth)


class MAE(PredictMetric):
    """
    Compute MAE (mean absolute error).  This is computed as:

    .. math::
        \\sum_{r_{ui} \\in R} \\left|r_{ui} - s(i|u)\\right|

    This metric does not do any fallbacks; if you want to compute MAE with
    fallback predictions (e.g. usign a bias model when a collaborative filter
    cannot predict), generate predictions with
    :class:`~lenskit.basic.FallbackScorer`.
    """

    @overload
    def __call__(self, predictions: ItemList, truth: ItemList | None = None) -> float: ...
    @overload
    def __call__(self, predictions: pd.DataFrame) -> float: ...
    def __call__(
        self, predictions: ItemList | pd.DataFrame, truth: ItemList | None = None
    ) -> float:
        return _score_predictions(_mae, predictions, truth, self.missing_scores, self.missing_truth)


def measure_user_predictions(
    predictions: pd.DataFrame, metric: PredictMetric | type[PredictMetric]
) -> pd.Series:
    """
    Compute per-user metrics for a set of predictions.

    Args:
        predictions:
            A data frame of predictions.  Must have `user_id`, `item_id`,
            `score`, and `rating` columns.
        metric:
            The metric to compute.  :fun:`rmse` and :fun:`mae` both implement
            this interface.
    """
    if isinstance(metric, type):
        metric = metric()

    return group_df(predictions).apply(lambda df: metric(df))
