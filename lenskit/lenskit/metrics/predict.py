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

from lenskit.data import ItemList, ItemListCollection
from lenskit.data.bulk import group_df
from lenskit.data.schemas import ITEM_COMPAT_COLUMN, normalize_columns
from lenskit.data.types import AliasedColumn

from ._base import GlobalMetric, ListMetric, Metric

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


class RMSE(PredictMetric, ListMetric, GlobalMetric):
    """
    Compute RMSE (root mean squared error).  This is computed as:

    .. math::
        \\sum_{r_{ui} \\in R} \\left(r_{ui} - s(i|u)\\right)^2

    This metric does not do any fallbacks; if you want to compute RMSE with
    fallback predictions (e.g. usign a bias model when a collaborative filter
    cannot predict), generate predictions with
    :class:`~lenskit.basic.FallbackScorer`.
    """

    @override
    def measure_list(self, predictions: ItemList, test: ItemList | None = None, /) -> float:
        ps, ts = self.align_scores(predictions, test)
        err = ps - ts
        err *= err
        return np.sqrt(np.mean(err))

    @override
    def measure_run(
        self, predictions: ItemListCollection, test: ItemListCollection | None = None, /
    ) -> float:
        sse = 0
        n = 0
        for key, plist in predictions:
            if test is None:
                tlist = None
            else:
                tlist = test.lookup_projected(key)
                if tlist is None:
                    _log.warning("missing truth for list %s", key)
                    if self.missing_truth == "error":
                        raise ValueError(f"missing truth for list {key}")
                    else:
                        continue

            ps, ts = self.align_scores(plist, tlist)
            err = ps - ts
            err *= err
            sse += np.sum(err)
            n += np.sum(ps.notna() & ts.notna())

        if n == 0:
            return np.nan
        else:
            return np.sqrt(sse / n)


class MAE(PredictMetric, ListMetric, GlobalMetric):
    """
    Compute MAE (mean absolute error).  This is computed as:

    .. math::
        \\sum_{r_{ui} \\in R} \\left|r_{ui} - s(i|u)\\right|

    This metric does not do any fallbacks; if you want to compute MAE with
    fallback predictions (e.g. usign a bias model when a collaborative filter
    cannot predict), generate predictions with
    :class:`~lenskit.basic.FallbackScorer`.
    """

    @override
    def measure_list(self, predictions: ItemList, test: ItemList | None = None, /) -> float:
        ps, ts = self.align_scores(predictions, test)
        err = ps - ts
        return np.mean(np.abs(err)).item()

    @override
    def measure_run(
        self, predictions: ItemListCollection, test: ItemListCollection | None = None, /
    ) -> float:
        sae = 0
        n = 0
        for key, plist in predictions:
            if test is None:
                tlist = None
            else:
                tlist = test.lookup_projected(key)
                if tlist is None:
                    _log.warning("missing truth for list %s", key)
                    if self.missing_truth == "error":
                        raise ValueError(f"missing truth for list {key}")
                    else:
                        continue

            ps, ts = self.align_scores(plist, tlist)
            err = ps - ts
            sae += np.sum(np.abs(err))
            n += np.sum(ps.notna() & ts.notna())

        if n == 0:
            return np.nan
        else:
            return sae / n


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

    return group_df(predictions).apply(lambda df: metric.measure_list(df))
