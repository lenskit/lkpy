import os
import os.path
import pathlib
import tempfile
import logging
import warnings

from joblib import Parallel, delayed, dump, load

import pandas as pd
import numpy as np

from ..algorithms import Recommender
from .. import util

_logger = logging.getLogger(__name__)


def _recommend_user(algo, user, n, candidates):
    _logger.debug('generating recommendations for %s', user)
    watch = util.Stopwatch()
    res = algo.recommend(user, n, candidates)
    _logger.debug('%s recommended %d/%d items for %s in %s', algo, len(res), n, user, watch)
    res['user'] = user
    res['rank'] = np.arange(1, len(res) + 1)
    return res


def __standard_cand_fun(candidates):
    """
    Convert candidates from the forms accepted by :py:fun:`recommend` into
    a standard form, a function that takes a user and returns a candidate
    list.
    """
    if isinstance(candidates, dict):
        return candidates.get
    elif candidates is None:
        return lambda u: None
    else:
        return candidates


def recommend(algo, users, n, candidates=None, *, nprocs=None, **kwargs):
    """
    Batch-recommend for multiple users.  The provided algorithm should be a
    :py:class:`algorithms.Recommender`.

    Args:
        algo: the algorithm
        users(array-like): the users to recommend for
        n(int): the number of recommendations to generate (None for unlimited)
        candidates:
            the users' candidate sets. This can be a function, in which case it will
            be passed each user ID; it can also be a dictionary, in which case user
            IDs will be looked up in it.  Pass ``None`` to use the recommender's
            built-in candidate selector (usually recommended).
        nprocs(int):
            The number of processes to use for parallel recommendations.  Passed as
            ``n_jobs`` to :cls:`joblib.Parallel`.  The default, ``None``, will make
            the process sequential _unless_ called inside the :func:`joblib.parallel_backend`
            context manager.

    Returns:
        A frame with at least the columns ``user``, ``rank``, and ``item``; possibly also
        ``score``, and any other columns returned by the recommender.
    """

    rec_algo = Recommender.adapt(algo)
    if candidates is None and rec_algo is not algo:
        warnings.warn('no candidates provided and algo is not a recommender, unlikely to work')
    del algo  # don't need reference any more

    if 'ratings' in kwargs:
        warnings.warn('Providing ratings to recommend is not supported', DeprecationWarning)

    candidates = __standard_cand_fun(candidates)

    loop = Parallel(n_jobs=nprocs)

    path = None
    try:
        with loop:
            backend = loop._backend.__class__.__name__
            njobs = loop._effective_n_jobs()
            _logger.info('parallel backend %s, effective njobs %s',
                         backend, njobs)
            if njobs > 1:
                fd, path = tempfile.mkstemp(prefix='lkpy-predict', suffix='.pkl')
                path = pathlib.Path(path)
                os.close(fd)
                _logger.debug('pre-serializing algorithm %s to %s', rec_algo, path)
                dump(rec_algo, path)
                rec_algo = load(path, mmap_mode='r')

            _logger.info('recommending for %d users (nprocs=%s)', len(users), nprocs)
            results = loop(delayed(_recommend_user)(rec_algo, user, n, candidates(user))
                           for user in users)

            del rec_algo

        results = pd.concat(results, ignore_index=True)
    finally:
        util.delete_sometime(path)

    return results
