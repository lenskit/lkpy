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
    if missing == 'error' and truth.isna().any():
        nmissing = truth.isna().sum()
        raise ValueError('missing truth for {} predictions'.format(nmissing))


def rmse(predictions, truth, missing='error'):
    """
    Compute RMSE (root mean squared error).

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
    predictions, truth = predictions.align(truth, join='left')
    _check_missing(truth, missing)

    diff = predictions - truth

    sqdiff = diff.apply(np.square)
    msq = sqdiff.mean()
    return np.sqrt(msq)


def mae(predictions, truth, missing='error'):
    """
    Compute MAE (mean absolute error).

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

    predictions, truth = predictions.align(truth, join='left')
    _check_missing(truth, missing)

    diff = predictions - truth

    adiff = diff.apply(np.abs)
    return adiff.mean()
