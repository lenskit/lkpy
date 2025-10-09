# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit test harnesses and utilities.

This package contains utility code for testing LensKit and its components,
including in derived packages.  It relies on PyTest and Hypothesis.
"""

import os
from contextlib import contextmanager

import numpy as np

import hypothesis.strategies as st

from ._arrays import coo_arrays, scored_lists, sparse_arrays, sparse_tensors
from ._components import BasicComponentTests, ScorerTests
from ._movielens import (
    DemoRecs,
    demo_recs,
    ml_20m,
    ml_100k,
    ml_100k_zip,
    ml_ds,
    ml_ds_unchecked,
    ml_ratings,
    ml_test_dir,
    pop_recs,
)

__all__ = [
    "coo_arrays",
    "scored_lists",
    "sparse_arrays",
    "sparse_tensors",
    "ml_100k",
    "ml_100k_zip",
    "ml_20m",
    "ml_ds",
    "ml_ds_unchecked",
    "ml_ratings",
    "ml_test_dir",
    "demo_recs",
    "pop_recs",
    "set_env_var",
    "DemoRecs",
    "BasicComponentTests",
    "ScorerTests",
]


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


def have_memory(gb: int | float) -> bool:
    "Check if we have at least gb gigs of memory."
    if not hasattr(os, "sysconf"):
        return False

    p_size = os.sysconf("SC_PAGE_SIZE")
    p_count = os.sysconf("SC_PHYS_PAGES")
    mem_size = p_size * p_count
    return mem_size >= gb * 1024 * 1024 * 1024


def integer_ids():
    """
    Hypothesis strategy to generate valid integer user/item IDs.
    """
    return st.integers(1, np.iinfo("i8").max)
