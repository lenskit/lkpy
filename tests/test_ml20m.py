"""
Tests on the ML-20M data set.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from lenskit.datasets import MovieLens
from lenskit import crossfold as xf
from lenskit.metrics import predict as pm
from lenskit import batch
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import Popular
from lenskit.algorithms.als import BiasedMF
from lenskit.algorithms import item_knn as knn

try:
    import lenskit_tf
except:
    lenskit_tf = None
from lenskit.util import Stopwatch
from lenskit.util import test as lktu

import pytest
from pytest import approx

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


@pytest.mark.realdata
@pytest.mark.slow
@pytest.mark.skip
@pytest.mark.skipif(
    lenskit_tf is None or not lenskit_tf.TF_AVAILABLE, reason="TensorFlow not available"
)
def test_tf_isvd(ml20m):
    algo = lenskit_tf.IntegratedBiasMF(20)

    def eval(train, test):
        _log.info("running training")
        algo.fit(train)
        _log.info("testing %d users", test.user.nunique())
        return batch.predict(algo, test)

    folds = xf.sample_users(ml20m, 2, 5000, xf.SampleFrac(0.2))
    preds = pd.concat(eval(train, test) for (train, test) in folds)
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.60, abs=0.025)

    user_rmse = preds.groupby("user").apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.92, abs=0.05)
