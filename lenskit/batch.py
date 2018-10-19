"""
Batch-run predictors and recommenders for evaluation.
"""

import logging
from functools import partial

import pandas as pd
import numpy as np

from .algorithms import Predictor, Recommender

_logger = logging.getLogger(__name__)


def predict(algo, pairs, model=None):
    """
    Generate predictions for user-item pairs.  The provided algorithm should be a
    :py:class:`algorithms.Predictor` or a function of two arguments: the user ID and
    a list of item IDs. It should return a dictionary or a :py:class:`pandas.Series`
    mapping item IDs to predictions.

    Args:
        predictor(callable or :py:class:algorithms.Predictor):
            a rating predictor function or algorithm.
        pairs(pandas.DataFrame):
            a data frame of (``user``, ``item``) pairs to predict for. If this frame also
            contains a ``rating`` column, it will be included in the result.
        model(any): a model for the algorithm.

    Returns:
        pandas.DataFrame:
            a frame with columns ``user``, ``item``, and ``prediction`` containing
            the prediction results. If ``pairs`` contains a `rating` column, this
            result will also contain a `rating` column.
    """

    if isinstance(algo, Predictor):
        pfun = partial(algo.predict, model)
    else:
        pfun = algo

    def run(user, udf):
        res = pfun(user, udf.item)
        return pd.DataFrame({'user': user, 'item': res.index, 'prediction': res.values})

    ures = (run(user, udf) for (user, udf) in pairs.groupby('user'))
    res = pd.concat(ures)
    if 'rating' in pairs:
        return pairs.join(res.set_index(['user', 'item']), on=('user', 'item'))
    return res


def recommend(algo, model, users, n, candidates, ratings=None):
    """
    Batch-recommend for multiple users.  The provided algorithm should be a
    :py:class:`algorithms.Recommender` or :py:class:`algorithms.Predictor` (which
    will be converted to a top-N recommender).

    Args:
        algo: the algorithm
        model: The algorithm model
        users(array-like): the users to recommend for
        n(int): the number of recommendations to generate (None for unlimited)
        candidates:
            the users' candidate sets. This can be a function, in which case it will
            be passed each user ID; it can also be a dictionary, in which case user
            IDs will be looked up in it.
        ratings(pandas.DataFrame):
            if not ``None``, a data frame of ratings to attach to recommendations when
            available.

    Returns:
        A frame with at least the columns ``user``, ``rank``, and ``item``; possibly also
        ``score``, and any other columns returned by the recommender.
    """

    if isinstance(candidates, dict):
        candidates = candidates.get
    algo = Recommender.adapt(algo)

    results = []
    for user in users:
        _logger.debug('generating recommendations for %s', user)
        ucand = candidates(user)
        res = algo.recommend(model, user, n, ucand)
        iddf = pd.DataFrame({'user': user, 'rank': np.arange(1, len(res) + 1)})
        results.append(pd.concat([iddf, res], axis='columns'))

    results = pd.concat(results, ignore_index=True)
    if ratings is not None:
        # combine with test ratings for relevance data
        results = pd.merge(results, ratings, how='left', on=('user', 'item'))
        # fill in missing 0s
        results.loc[results.rating.isna(), 'rating'] = 0

    return results
