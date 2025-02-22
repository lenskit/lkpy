# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit test harnesses and utilities.

This package contains utility code for testing LensKit and its components,
including in derived packages.  It relies on PyTest and Hypothesis.
"""

import os
from contextlib import contextmanager

from ._arrays import coo_arrays, scored_lists, sparse_arrays, sparse_tensors
from ._components import BasicComponentTests, ScorerTests
from ._markers import jit_enabled, wantjit
from ._movielens import (
    DemoRecs,
    demo_recs,
    ml_100k,
    ml_100k_zip,
    ml_ds,
    ml_ds_unchecked,
    ml_ratings,
    ml_test_dir,
)

__all__ = [
    "coo_arrays",
    "scored_lists",
    "sparse_arrays",
    "sparse_tensors",
    "ml_100k",
    "ml_100k_zip",
    "ml_ds",
    "ml_ds_unchecked",
    "ml_ratings",
    "ml_test_dir",
    "demo_recs",
    "wantjit",
    "jit_enabled",
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
