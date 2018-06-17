"""
Batch-run predictors and recommenders for evaluation.
"""

import logging

import pandas as pd

_logger = logging.getLogger(__package__)


def predict(algo, model, pairs):
    """
    Generate prediction for user-item pairs.
    """

    _logger.debug('running with frame\n%s', pairs)

    ures = (algo.predict(model, user, udf.item).reset_index(name='prediction').assign(user=user)
            for (user, udf) in pairs.groupby('user'))
    res = pd.concat(ures).loc[:, ['user', 'item', 'prediction']]
    _logger.debug('returning frame\n%s', res)
    return res
