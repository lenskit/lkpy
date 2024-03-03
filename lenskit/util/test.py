# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Test utilities for LKPY tests.
"""

import os
import os.path
from contextlib import contextmanager

import pytest

from lenskit.algorithms.basic import PopScore
from lenskit.algorithms.ranking import PlackettLuce
from lenskit.batch import recommend
from lenskit.crossfold import simple_test_pair
from lenskit.datasets import ML100K, MovieLens

ml_test = MovieLens("data/ml-latest-small")
ml100k = ML100K("data/ml-100k")


@pytest.fixture(scope="session")
def demo_recs():
    """
    A demo set of train, test, and recommendation data.
    """
    train, test = simple_test_pair(ml_test.ratings, f_rates=0.5)

    users = test["user"].unique()
    algo = PopScore()
    algo = PlackettLuce(algo, rng_spec="user")
    algo.fit(train)

    recs = recommend(algo, users, 500)
    return train, test, recs


@contextmanager
def set_env_var(var, val):
    "Set an environment variable & restore it."
    is_set = var in os.environ
    old_val = None
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


wantjit = pytest.mark.skipif("NUMBA_DISABLE_JIT" in os.environ, reason="JIT required")
