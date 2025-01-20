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

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import Callable, Literal, TypeAlias, override

from lenskit.data import ItemList
from lenskit.data.adapt import ITEM_COMPAT_COLUMN, normalize_columns
from lenskit.data.types import AliasedColumn

from ._base import DecomposedMetric, ListMetric, Metric

_log = logging.getLogger(__name__)

MissingDisposition: TypeAlias = Literal["error", "ignore"]
ScoreArray: TypeAlias = NDArray[np.floating] | pd.Series
PredMetric: TypeAlias = Callable[[ScoreArray, ScoreArray], float]


class PredictMetric(Metric):
    """
    Extension to the metric function interface for prediction metrics.

    In addition to the general metric interface, predict metrics can be called
    with a single item list (or item list collection) that has both ``scores``
    and a ``rating`` field.

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

    Stability:
        Caller
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

    def align_scores(
        self, predictions: ItemList, truth: ItemList | None = None
    ) -> tuple[pd.Series[float], pd.Series[float]]:
        """
        Align prediction scores and rating values, applying the configured
        missing dispositions.  The result is two Pandas series, predictions and
        truth, that are aligned and checked for missing data in accordance with
        the configured options.
        """
        if isinstance(predictions, pd.DataFrame):
            df = normalize_columns(
                predictions, ITEM_COMPAT_COLUMN, AliasedColumn("score", ["prediction"])
            )
            predictions = ItemList.from_df(df)

        if not isinstance(predictions, ItemList):  # pragma: nocover
            raise TypeError(f"predictions must be ItemList, not {type(predictions)}")
        if truth is not None and not isinstance(truth, ItemList):  # pragma: nocover
            raise TypeError(f"truth must be ItemList, not {type(truth)}")

        pred_s = predictions.scores("pandas", index="ids")
        assert pred_s is not None, "item list does not have scores"
        if truth is not None:
            rate_s = truth.field("rating", "pandas", index="ids")
        else:
            rate_s = predictions.field("rating", "pandas", index="ids")
        assert rate_s is not None, "no ratings provided"
        pred_s, rate_s = pred_s.align(rate_s, join="outer")

        pred_m = pred_s.isna()
        rate_m = rate_s.isna()

        if self.missing_scores == "error" and (nbad := np.sum(pred_m & ~rate_m)):
            raise ValueError(f"missing scores for {nbad} truth items")

        if self.missing_truth == "error" and (nbad := np.sum(rate_m & ~pred_m)):
            raise ValueError(f"missing truth for {nbad} scored items")

        return pred_s, rate_s


class RMSE(PredictMetric, ListMetric, DecomposedMetric):
    """
    Compute RMSE (root mean squared error).  This is computed as:

    .. math::
        \\sum_{r_{ui} \\in R} \\left(r_{ui} - s(i|u)\\right)^2

    This metric does not do any fallbacks; if you want to compute RMSE with
    fallback predictions (e.g. usign a bias model when a collaborative filter
    cannot predict), generate predictions with
    :class:`~lenskit.basic.FallbackScorer`.

    Stability:
        Caller
    """

    @override
    def measure_list(self, predictions: ItemList, test: ItemList | None = None, /) -> float:
        ps, ts = self.align_scores(predictions, test)
        err = ps - ts
        err *= err
        return np.sqrt(np.mean(err))

    @override
    def compute_list_data(self, output, test):
        ps, ts = self.align_scores(output, test)
        err = ps - ts
        err *= err
        return np.sum(err), len(err)

    @override
    def extract_list_metric(self, metric):
        tot, n = metric
        return np.sqrt(tot / n)

    @override
    def global_aggregate(self, values):
        tot_sqerr = 0.0
        tot_n = 0.0
        for t, n in values:
            tot_sqerr += t
            tot_n += n

        return np.sqrt(tot_sqerr / tot_n)


class MAE(PredictMetric, ListMetric, DecomposedMetric):
    """
    Compute MAE (mean absolute error).  This is computed as:

    .. math::
        \\sum_{r_{ui} \\in R} \\left|r_{ui} - s(i|u)\\right|

    This metric does not do any fallbacks; if you want to compute MAE with
    fallback predictions (e.g. usign a bias model when a collaborative filter
    cannot predict), generate predictions with
    :class:`~lenskit.basic.FallbackScorer`.

    Stability:
        Caller
    """

    @override
    def measure_list(self, predictions: ItemList, test: ItemList | None = None, /) -> float:
        ps, ts = self.align_scores(predictions, test)
        err = ps - ts
        return np.mean(np.abs(err)).item()

    @override
    def compute_list_data(self, output, test):
        ps, ts = self.align_scores(output, test)
        err = ps - ts
        return np.sum(np.abs(err)), len(err)

    @override
    def extract_list_metric(self, metric):
        tot, n = metric
        return tot / n

    @override
    def global_aggregate(self, values):
        tot_err = 0.0
        tot_n = 0.0
        for t, n in values:
            tot_err += t
            tot_n += n

        return tot_err / tot_n
