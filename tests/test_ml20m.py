# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests on the ML-20M data set.
"""

import logging
from pathlib import Path

import pytest

from lenskit import batch
from lenskit.algorithms import Recommender
from lenskit.algorithms.als import BiasedMF
from lenskit.algorithms.basic import Popular
from lenskit.datasets import MovieLens

_log = logging.getLogger(__name__)

_ml_path = Path("data/ml-20m")
if _ml_path.exists():
    _ml_20m = MovieLens(_ml_path)
else:
    _ml_20m = None


@pytest.fixture
def ml20m():
    if _ml_20m:
        return _ml_20m.ratings
    else:
        pytest.skip("ML-20M not available")


@pytest.mark.slow
@pytest.mark.realdata
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_pop_recommend(ml20m, rng, n_jobs):
    users = rng.choice(ml20m["user"].unique(), 10000, replace=False)
    algo = Popular()
    _log.info("training %s", algo)
    algo.fit(ml20m)
    _log.info("recommending with %s", algo)
    recs = batch.recommend(algo, users, 10, n_jobs=n_jobs)

    assert recs["user"].nunique() == 10000
