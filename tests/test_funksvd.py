import os
import logging

import lenskit.algorithms.funksvd as svd

import pandas as pd
import numpy as np

import pytest
from pytest import approx, mark

import lk_test_utils as lktu

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


def test_fsvd_basic_build():
    algo = svd.FunkSVD(20, iterations=20)
    model = algo.train(simple_df)

    assert model is not None
    assert model.global_bias == approx(simple_df.rating.mean())


def test_fsvd_clamp_build():
    algo = svd.FunkSVD(20, iterations=20, range=(1, 5))
    model = algo.train(simple_df)

    assert model is not None
    assert model.global_bias == approx(simple_df.rating.mean())


@mark.slow
@mark.eval
def test_fsvd_batch_accuracy():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    if not os.path.exists('ml-100k/u.data'):
        raise pytest.skip()

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    svd_algo = svd.FunkSVD(25, 125, damping=10)
    algo = basic.Fallback(svd_algo, basic.Bias(damping=10))

    def eval(train, test):
        _log.info('running training')
        model = algo.train(train)
        _log.info('testing %d users', test.user.nunique())
        return batch.predict(lambda u, xs: algo.predict(model, u, xs), test)

    preds = batch.multi_predict(xf.partition_users(ratings, 5, xf.SampleFrac(0.2)),
                                algo, processes=1)
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.74, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.92, abs=0.05)
