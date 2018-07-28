"""
Batch-run predictors and recommenders for evaluation.
"""

import os
import logging
import multiprocessing
from functools import partial
from collections import namedtuple

try:
    from multiprocessing_logging import install_mp_handler
    install_mp_handler()
except ImportError:
    pass

import pandas as pd
import numpy as np

from .algorithms import Predictor, Recommender

_logger = logging.getLogger(__package__)


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

    ures = (pfun(user, udf.item).reset_index(name='prediction').assign(user=user)
            for (user, udf) in pairs.groupby('user'))
    res = pd.concat(ures).loc[:, ['user', 'item', 'prediction']]
    if 'rating' in pairs:
        return pairs.join(res.set_index(['user', 'item']), on=('user', 'item'))
    return res


_MPState = namedtuple('_MPState', ['train', 'test', 'algo'])


def _run_mpjob(job: _MPState) -> bytes:
    train = pd.read_msgpack(job.train)
    _logger.info('training %s on %d rows', job.algo, len(train))
    model = job.algo.train(train)
    test = pd.read_msgpack(job.test)
    _logger.info('generating predictions with %s for %d pairs', job.algo, len(test))
    results = predict(job.algo, test, model)
    return results.to_msgpack()


def _mp_stateify(sets, algo):
    for train, test in sets:
        train_bytes = train.to_msgpack()
        test_bytes = test.to_msgpack()
        yield _MPState(train_bytes, test_bytes, algo)


def _run_spjob(algo, train, test):
    _logger.info('training %s on %d rows', algo, len(train))
    model = algo.train(train)
    _logger.info('generating predictions with %s for %d pairs', algo, len(test))
    results = predict(algo, test, model)
    return results


def multi_predict(sets, algo, processes=None):
    _logger.info('launching multi-predict with %s processes', processes)
    if processes is None or processes > 1:
        if processes is None and 'LK_PROCESS_COUNT' in os.environ:
            processes = int(os.environ['LK_PROCESS_COUNT'])
        with multiprocessing.Pool(processes) as p:
            results = [pd.read_msgpack(rbs) for rbs in p.map(_run_mpjob, _mp_stateify(sets, algo))]
    else:
        results = [_run_spjob(algo, train, test) for train, test in sets]

    _logger.info('finished %d predict jobs', len(results))

    return pd.concat(results)


def recommend(algo, model, users, n, candidates):
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

    Returns:
        A frame with at least the columns ``user``, ``rank``, and ``item``; possibly also
        ``score``, and any other columns returned by the recommender.
    """

    if isinstance(candidates, dict):
        candidates = candidates.get
    algo = Recommender.adapt(algo)

    results = []
    for user in users:
        ucand = candidates(user)
        res = algo.recommend(model, user, n, ucand)
        iddf = pd.DataFrame({'user': user, 'rank': np.arange(1, len(res) + 1)})
        results.append(pd.concat([iddf, res], axis='columns'))

    return pd.concat(results, ignore_index=True)
