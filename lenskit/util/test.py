"""
Test utilities for LKPY tests.
"""

import os
import os.path
import logging

import numpy as np
from .. import matrix

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


def rand_csr(nrows=100, ncols=50, nnz=1000, values=True):
    "Generate a random CSR for testing."
    coords = np.random.choice(np.arange(ncols * nrows, dtype=np.int32), nnz, False)
    rows = np.mod(coords, nrows, dtype=np.int32)
    cols = np.floor_divide(coords, nrows, dtype=np.int32)
    if values:
        vals = np.random.randn(nnz)
    else:
        vals = None
    return matrix.CSR.from_coo(rows, cols, vals, (nrows, ncols))


wantjit = pytest.mark.skipif('NUMBA_DISABLE_JIT' in os.environ,
                             reason='JIT required')
