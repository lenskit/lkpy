"""
Test utilities for LKPY tests.
"""

import os
import os.path
import logging
from contextlib import contextmanager

import numpy as np
from .. import matrix

import pytest

from lenskit.datasets import MovieLens, ML100K

_log = logging.getLogger(__name__)

ml_test = MovieLens('data/ml-latest-small')
ml100k = ML100K('data/ml-100k')


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


@contextmanager
def set_env_var(var, val):
    "Set an environment variable & restore it."
    is_set = var in os.environ
    if is_set:
        old_val = os.environ[var]
    try:
        if val is None:
            if is_set:
                del os.environ[var]
        else:
            os.environ[var] = val
        yield
    finally:
        if is_set:
            os.environ[var] = old_val
        elif val is not None:
            del os.environ[var]


wantjit = pytest.mark.skipif('NUMBA_DISABLE_JIT' in os.environ,
                             reason='JIT required')
