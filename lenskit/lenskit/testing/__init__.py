"""
LensKit test harnesses and utilities.

This package contains utility code for testing LensKit and its components,
including in derived packages.  It relies on PyTest and Hypothesis.
"""

import os
from contextlib import contextmanager

import pytest

from .arrays import coo_arrays, scored_lists, sparse_arrays, sparse_tensors
from .movielens import demo_recs, ml_100k, ml_100k_zip, ml_ds, ml_ratings, ml_test_dir

__all__ = [
    "coo_arrays",
    "scored_lists",
    "sparse_arrays",
    "sparse_tensors",
    "ml_100k",
    "ml_100k_zip",
    "ml_ds",
    "ml_ratings",
    "ml_test_dir",
    "demo_recs",
    "wantjit",
    "jit_enabled",
    "set_env_var",
]

jit_enabled = True
if "NUMBA_DISABLE_JIT" in os.environ:
    jit_enabled = False
if os.environ.get("PYTORCH_JIT", None) == "0":
    jit_enabled = False

wantjit = pytest.mark.skipif(not jit_enabled, reason="JIT required")


@contextmanager
def set_env_var(var, val):
    "Set an environment variable & restore it."
    old_val = os.environ.get(var, None)
    try:
        if val is None:
            if old_val is not None:
                del os.environ[var]
        else:
            os.environ[var] = val
        yield
    finally:
        if old_val is not None:
            os.environ[var] = old_val
        elif val is not None:
            del os.environ[var]
