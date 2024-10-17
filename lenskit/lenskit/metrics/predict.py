# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Prediction accuracy metrics.
"""

from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

from lenskit.data import ItemList

MissingDisposition: TypeAlias = Literal["error", "ignore"]


def _check_missing(truth: pd.Series, missing: MissingDisposition):
    """
    Check for missing truth values.

    Args:
        truth:
            the series of truth values
        missing:
            what to do with missing values
    """
    nmissing = truth.isna().sum()

    if missing == "error" and nmissing:
        raise ValueError("missing truth for {} predictions".format(nmissing))


def rmse(predictions: ItemList, truth: ItemList, missing: MissingDisposition = "error") -> float:
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

    Args:
        predictions:
            item list with scored items.
        truth:
            ground truth ratings from data
        missing:
            how to handle predictions without truth.

    Returns:
        the root mean squared approximation error
    """

    pred_s = predictions.scores("pandas", index="ids")
    assert pred_s is not None, "predictions have no scores"
    rate_s = truth.field("rating", "pandas", index="ids")
    assert rate_s is not None, "truth has no ratings"

    # realign
    pred_s, rate_s = pred_s.align(rate_s, join="left")
    _check_missing(rate_s, missing)

    diff = pred_s - rate_s

    sqdiff = np.square(diff)
    msq = sqdiff.mean()
    return np.sqrt(msq)


def mae(predictions: ItemList, truth: ItemList, missing: MissingDisposition = "error"):
    """
    Compute MAE (mean absolute error).  This is computed as:

    .. math::
        \\sum_{r_{ui} \\in R} \\left|r_{ui} - s(i|u)\\right|

    This computes *per-user* MAE. It does not do any fallbacks; if you want to
    compute MAE with fallback predictions (e.g. usign a bias model when a
    collaborative filter cannot predict), generate predictions with
    :class:`~lenskit.basic.FallbackScorer`.

    Args:
        predictions:
            the predictions
        truth:
            the ground truth ratings from data
        missing:
            how to handle predictions without truth. Can be one of ``'error'``
            or ``'ignore'``.

    Returns:
        double: the mean absolute approximation error
    """

    pred_s = predictions.scores("pandas", index="ids")
    assert pred_s is not None, "predictions have no scores"
    rate_s = truth.field("rating", "pandas", index="ids")
    assert rate_s is not None, "truth has no ratings"

    # realign
    pred_s, rate_s = pred_s.align(rate_s, join="left")
    _check_missing(rate_s, missing)

    diff = pred_s - rate_s

    adiff = np.abs(diff)
    return adiff.mean()
