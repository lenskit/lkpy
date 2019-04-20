"""
Test utilities for LKPY tests.
"""

import os
import os.path
import logging

import pytest

from lenskit.datasets import MovieLens, ML100K

_log = logging.getLogger(__name__)

ml_test = MovieLens('ml-latest-small')
ml100k = ML100K()


def ml_sample():
    ratings = ml_test.ratings
    icounts = ratings.groupby('item').rating.count()
    top = icounts.nlargest(500)
    ratings = ratings.set_index('item')
    top_rates = ratings.loc[top.index, :]
    _log.info('top 500 items yield %d of %d ratings', len(top_rates), len(ratings))
    return top_rates.reset_index()


wantjit = pytest.mark.skipif('NUMBA_DISABLE_JIT' in os.environ,
                             reason='JIT required')
