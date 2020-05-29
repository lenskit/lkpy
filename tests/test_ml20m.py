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

import pytest

_log = logging.getLogger(__name__)

_ml_path = Path('ml-20m')
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


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_pop_recommend(ml20m, n_jobs):
    all_test = []
    all_recs = []
    for train, test in xf.sample_users(ml20m, 5, 5000, xf.SampleN(5)):
        algo = Popular()
        _log.info('training %s', algo)
        algo.fit(train)
        _log.info('recommending with %s', algo)
        recs = batch.recommend(algo, test['user'].unique(), 10, n_jobs=n_jobs)
        all_recs.append(recs)
        all_test.append(test)

    all_recs = pd.concat(all_recs)
    assert all_recs['user'].nunique() == 25000
