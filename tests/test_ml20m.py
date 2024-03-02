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


from lenskit.datasets import MovieLens
from lenskit import batch
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import Popular
from lenskit.algorithms.als import BiasedMF

import pytest

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


@pytest.mark.realdata
@pytest.mark.slow
def test_als_isolate(ml20m, rng):
    users = rng.choice(ml20m["user"].unique(), 5000, replace=False)
    algo = BiasedMF(20, iterations=10)
    algo = Recommender.adapt(algo)
    _log.info("training %s", algo)
    ares = batch.train_isolated(algo, ml20m)
    try:
        _log.info("recommending with %s", algo)
        recs = batch.recommend(ares, users, 10)
        assert recs["user"].nunique() == 5000
        _log.info("predicting with %s", algo)
        pairs = ml20m.sample(1000)
        preds = batch.predict(ares, pairs)
        assert len(preds) == len(pairs)
    finally:
        ares.close()
