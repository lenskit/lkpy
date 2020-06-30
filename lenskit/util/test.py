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
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

from lenskit.datasets import MovieLens, ML100K

_log = logging.getLogger(__name__)

ml_test = MovieLens('data/ml-latest-small')
ml100k = ML100K('data/ml-100k')


@st.composite
def csrs(draw, nrows=None, ncols=None, nnz=None, values=None):
    if ncols is None:
        ncols = draw(st.integers(5, 100))
    elif not isinstance(ncols, int):
        ncols = draw(ncols)

    if nrows is None:
        nrows = draw(st.integers(5, 100))
    elif not isinstance(nrows, int):
        nrows = draw(nrows)

    if nnz is None:
        nnz = draw(st.integers(10, nrows * ncols // 2))
    elif not isinstance(nnz, int):
        nnz = draw(nnz)

    coords = draw(nph.arrays(np.int32, nnz, elements=st.integers(0, nrows*ncols - 1), unique=True))
    rows = np.mod(coords, nrows, dtype=np.int32)
    cols = np.floor_divide(coords, nrows, dtype=np.int32)
    if values is None:
        values = draw(st.booleans())
    if values:
        rng = draw(st.randoms())
        vals = np.array([rng.normalvariate(0, 1) for i in range(nnz)])
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
