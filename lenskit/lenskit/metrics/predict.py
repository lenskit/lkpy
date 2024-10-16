# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Prediction accuracy metrics.
"""

import numpy as np
import pandas as pd


def _check_missing(truth, missing):
    """
    Check for missing truth values.

    Args:
        truth: the series of truth values
        missing: what to do with missing values
    """
    if missing == "error" and truth.isna().any():
        nmissing = truth.isna().sum()
        raise ValueError("missing truth for {} predictions".format(nmissing))


def rmse(predictions, truth, missing="error"):
    """
    Compute RMSE (root mean squared error).  This is computed as:

    .. math::
        \\sum_{r_{ui} \\in R} \\left(r_{ui} - s(i|u)\\right)^2

    When used with :func:`user_metric`, or on series grouped by user, it computes
    a per-user RMSE; when applied to an entire prediction frame, it computes global
    RMSE.  It does not do any fallbacks; if you want to compute RMSE with fallback
    predictions (e.g. usign a bias model when a collaborative filter cannot predict),
    generate predictions with :class:`lenskit.algorithms.basic.Fallback`.

    Args:
        predictions(pandas.Series): the predictions
        truth(pandas.Series): the ground truth ratings from data
        missing(string):
            how to handle predictions without truth. Can be one of
            ``'error'`` or ``'ignore'``.

    Returns:
        double: the root mean squared approximation error
    """

    # force into series (basically no-op if already a series)
    predictions = pd.Series(predictions)
    truth = pd.Series(truth)

    # realign
    predictions, truth = predictions.align(truth, join="left")
    _check_missing(truth, missing)

    diff = predictions - truth

    sqdiff = diff.apply(np.square)
    msq = sqdiff.mean()
    return np.sqrt(msq)


def mae(predictions, truth, missing="error"):
    """
    Compute MAE (mean absolute error).  This is computed as:

    .. math::
        \\sum_{r_{ui} \\in R} \\left|r_{ui} - s(i|u)\\right|

    When used with :func:`user_metric`, or on series grouped by user, it computes
    a per-user MAE; when applied to an entire prediction frame, it computes global
    MAE.  It does not do any fallbacks; if you want to compute MAE with fallback
    predictions (e.g. usign a bias model when a collaborative filter cannot predict),
    generate predictions with :class:`lenskit.algorithms.basic.Fallback`.

    Args:
        predictions(pandas.Series): the predictions
        truth(pandas.Series): the ground truth ratings from data
        missing(string):
            how to handle predictions without truth. Can be one of
            ``'error'`` or ``'ignore'``.

    Returns:
        double: the mean absolute approximation error
    """

    # force into series
    predictions = pd.Series(predictions)
    truth = pd.Series(truth)

    predictions, truth = predictions.align(truth, join="left")
    _check_missing(truth, missing)

    diff = predictions - truth

    adiff = diff.apply(np.abs)
    return adiff.mean()
