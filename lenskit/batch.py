"""
Batch-run predictors and recommenders for evaluation.
"""

import logging
import typing as typ

import pandas as pd

_logger = logging.getLogger(__package__)


def predict(predictor, pairs):
    """
    Generate predictions for user-item pairs.  The provided predictor should be a
    function of two arguments: the user ID and a list of item IDs. It should return
    a dictionary or a :py:class:`pandas.Series` mapping item IDs to predictions.

    Args:
        predictor(callable): a rating predictor function.
        model(any): a model for the algorithm.
        pairs(pandas.DataFrame): a data frame of (``user``, ``item``) pairs to predict for.

    Returns:
        pandas.DataFrame: a frame with columns ``user``, ``item``, and ``prediction`` containing
                          the prediction results.
    """

    _logger.debug('running with frame\n%s', pairs)

    ures = (predictor(user, udf.item).reset_index(name='prediction').assign(user=user)
            for (user, udf) in pairs.groupby('user'))
    res = pd.concat(ures).loc[:, ['user', 'item', 'prediction']]
    _logger.debug('returning frame\n%s', res)
    return res
