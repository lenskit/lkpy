# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Prediction accuracy metrics.
"""

import warnings
from typing import Callable, Literal, Protocol, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lenskit.data import ItemList

MissingDisposition: TypeAlias = Literal["error", "ignore"]
ScoreArray: TypeAlias = NDArray[np.floating] | pd.Series
PredMetric: TypeAlias = Callable[[ScoreArray, ScoreArray], float]


class PredictMetric(Protocol):
    """
    Interface implemented by prediction metrics.
    """

    def __call__(
        self,
        predictions: ItemList | pd.DataFrame,
        truth: ItemList | None = None,
        missing_scores: MissingDisposition = "error",
        missing_truth: MissingDisposition = "error",
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


def rmse(
    predictions: ItemList | pd.DataFrame,
    truth: ItemList | None = None,
    missing_scores: MissingDisposition = "error",
    missing_truth: MissingDisposition = "error",
) -> float:
    """
    Compute RMSE (root mean squared error).  This is computed as:

    .. math::
        \\sum_{r_{ui} \\in R} \\left(r_{ui} - s(i|u)\\right)^2

    This computes *per-user* RMSE: given an :class:`ItemList` containing the
    scores for items for a user, and another containing ratings (in the `rating`
    field), it will compute the RMSE for that user's predictions. Alternatively,
    the ground-truth ratings can be provided as a `rating` field on the scored
    item list.

    This metric does not do any fallbacks; if you want to compute RMSE with
    fallback predictions (e.g. usign a bias model when a collaborative filter
    cannot predict), generate predictions with
    :class:`~lenskit.basic.FallbackScorer`.

    If ``predictions`` is a data frame with scores for multiple users, this
    computes the global (micro-averaged) RMSE.  If it is a data frame or item
    list with a single user's scores, it computes the RMSE for that user.

    Args:
        predictions:
            Item list or data frame with scored items, optionally with ratings
            (if ``truth=None``).
        truth:
            Ground truth ratings from data.
        missing_scores:
            How to handle truth items without scores or predictions.
        missing_truth:
            How to handle predictions without truth.

    Returns:
        the root mean squared approximation error
    """

    return _score_predictions(_rmse, predictions, truth, missing_scores, missing_truth)


def mae(
    predictions: ItemList | pd.DataFrame,
    truth: ItemList | None = None,
    missing_scores: MissingDisposition = "error",
    missing_truth: MissingDisposition = "error",
) -> float:
    """
    Compute MAE (mean absolute error).  This is computed as:

    .. math::
        \\sum_{r_{ui} \\in R} \\left|r_{ui} - s(i|u)\\right|

    This computes *per-user* MAE. It does not do any fallbacks; if you want to
    compute MAE with fallback predictions (e.g. usign a bias model when a
    collaborative filter cannot predict), generate predictions with
    :class:`~lenskit.basic.FallbackScorer`.

    If ``predictions`` is a data frame with scores for multiple users, this
    computes the global (micro-averaged) MAE.  If it is a data frame or item
    list with a single user's scores, it computes the MAE for that user.

    Args:
        predictions:
            Item list or data frame with scored items, optionally with ratings
            (if ``truth=None``).
        truth:
            Ground truth ratings from data.
        missing_scores:
            How to handle truth items without scores or predictions.
        missing_truth:
            How to handle predictions without truth.

    Returns:
        double: the mean absolute approximation error
    """

    return _score_predictions(_mae, predictions, truth, missing_scores, missing_truth)


def measure_user_predictions(
    predictions: pd.DataFrame,
    metric: PredictMetric,
    missing_scores: MissingDisposition = "error",
    missing_truth: MissingDisposition = "error",
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

    if "user_id" not in predictions.columns:
        if "user_id" in predictions.index.names:
            predictions = predictions.reset_index("user_id")
        elif "user" in predictions.columns:
            warnings.warn("predictions has legacy “user” column")
            predictions = predictions.rename(columns={"user": "user_id"})
        elif "user" in predictions.index.names:
            warnings.warn("predictions has legacy “user” index name")
            predictions = predictions.reset_index("user").rename(columns={"user": "user_id"})
        else:
            raise KeyError("predictions has no user_id column")

    return predictions.groupby("user_id").apply(
        lambda df: metric(df, missing_scores=missing_scores, missing_truth=missing_truth)
    )
