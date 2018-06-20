"""
Prediction accuracy metrics.
"""

import numpy as np
import pandas as pd


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

    # force into series
    predictions = pd.Series(predictions)
    truth = pd.Series(truth)

    predictions, truth = predictions.align(truth, join='left')
    if missing == 'error' and truth.isna().any():
        nmissing = truth.isna().sum()
        raise ValueError('missing truth for ${nmissing} predictions')

    diff = predictions - truth

    sqdiff = diff.apply(np.square)
    msq = sqdiff.mean()
    return np.sqrt(msq)
