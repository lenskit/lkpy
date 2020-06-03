"""
Tests on the ML-20M data set.
"""

import logging
from pathlib import Path

import pandas as pd

from lenskit.datasets import MovieLens
from lenskit import crossfold as xf
from lenskit import batch
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import Popular
from lenskit.util.test import rng

import pytest

_log = logging.getLogger(__name__)

_ml_path = Path('data/ml-20m')
if _ml_path.exists():
    _ml_20m = MovieLens(_ml_path)
else:
    _ml_20m = None


@pytest.fixture
def ml20m():
    if _ml_20m:
        return _ml_20m.ratings
    else:
        pytest.skip('ML-20M not available')


@pytest.mark.slow
@pytest.mark.parametrize('n_jobs', [1, 2])
def test_pop_recommend(ml20m, rng, n_jobs):
    users = rng.choice(ml20m['user'].unique(), 10000, replace=False)
    algo = Popular()
    _log.info('training %s', algo)
    algo.fit(ml20m)
    _log.info('recommending with %s', algo)
    recs = batch.recommend(algo, users, 10, n_jobs=n_jobs)

    assert recs['user'].nunique() == 10000
