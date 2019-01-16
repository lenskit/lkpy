import logging
import multiprocessing as mp
from multiprocessing.pool import Pool

import pandas as pd
import numpy as np

from ..algorithms import Recommender
from .. import util

_logger = logging.getLogger(__name__)
_rec_context = None


class MPRecContext:
    def __init__(self, algo, candidates, size):
        self.algo = algo
        self.candidates = candidates
        self.size = size

    def __enter__(self):
        global _rec_context
        _logger.debug('installing context for %s', self.algo)
        _rec_context = self
        return self

    def __exit__(self, *args, **kwargs):
        global _rec_context
        _logger.debug('uninstalling context for %s', self.algo)
        _rec_context = None


def _recommend_user(algo, user, n, candidates):
    _logger.debug('generating recommendations for %s', user)
    watch = util.Stopwatch()
    res = algo.recommend(user, n, candidates)
    _logger.debug('%s recommended %d/%d items for %s in %s', algo, len(res), n, user, watch)
    res['user'] = user
    res['rank'] = np.arange(1, len(res) + 1)
    return res


def _recommend_seq(algo, users, n, candidates):
    if isinstance(candidates, dict):
        candidates = candidates.get
    algo = Recommender.adapt(algo)
    results = [_recommend_user(algo, user, n, candidates(user))
               for user in users]
    return results


def _recommend_worker(user):
    candidates = _rec_context.candidates(user)
    algo = Recommender.adapt(_rec_context.algo)
    res = _recommend_user(algo, user, _rec_context.size, candidates)
    return res.to_msgpack()


def recommend(algo, users, n, candidates, ratings=None, nprocs=None):
    """
    Batch-recommend for multiple users.  The provided algorithm should be a
    :py:class:`algorithms.Recommender` or :py:class:`algorithms.Predictor` (which
    will be converted to a top-N recommender).

    Args:
        algo: the algorithm
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

    if nprocs and nprocs > 1 and mp.get_start_method() == 'fork':
        _logger.info('starting recommend process with %d workers', nprocs)
        with MPRecContext(algo, candidates, n), Pool(nprocs) as pool:
            results = pool.map(_recommend_worker, users)
        results = [pd.read_msgpack(r) for r in results]
    else:
        _logger.info('starting sequential recommend process')
        results = _recommend_seq(algo, users, n, candidates)

    results = pd.concat(results, ignore_index=True)

    if ratings is not None and 'rating' in ratings.columns:
        # combine with test ratings for relevance data
        results = pd.merge(results, ratings, how='left', on=('user', 'item'))
        # fill in missing 0s
        results.loc[results.rating.isna(), 'rating'] = 0

    return results
