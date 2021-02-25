import logging
import pickle

from lenskit.algorithms import svd

import pandas as pd
import numpy as np

from pytest import approx, mark

import lenskit.util.test as lktu
from lenskit.util import clone

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})

need_skl = mark.skipif(not svd.SKL_AVAILABLE, reason='scikit-learn not installed')


@need_skl
def test_svd_basic_build():
    algo = svd.BiasedSVD(2)
    algo.fit(simple_df)

    assert algo.user_components_.shape == (3, 2)


@need_skl
def test_svd_predict_basic():
    algo = svd.BiasedSVD(2)
    algo.fit(simple_df)

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3] <= 5


@need_skl
def test_svd_predict_bad_item():
    algo = svd.BiasedSVD(2)
    algo.fit(simple_df)

    preds = algo.predict_for_user(10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


@need_skl
def test_svd_predict_bad_user():
    algo = svd.BiasedSVD(2)
    algo.fit(simple_df)

    preds = algo.predict_for_user(50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@need_skl
def test_svd_clone():
    algo = svd.BiasedSVD(5, damping=10)

    a2 = clone(algo)
    assert a2.factorization.n_components == algo.factorization.n_components
    assert a2.bias.user_damping == algo.bias.user_damping
    assert a2.bias.item_damping == algo.bias.item_damping

@need_skl
@mark.slow
def test_svd_save_load():
    ratings = lktu.ml_test.ratings

    original = svd.BiasedSVD(20)
    original.fit(ratings)

    mod = pickle.dumps(original)
    _log.info('serialized to %d bytes', len(mod))
    algo = pickle.loads(mod)

    assert algo.bias.mean_ == original.bias.mean_
    assert np.all(algo.bias.user_offsets_ == original.bias.user_offsets_)
    assert np.all(algo.bias.item_offsets_ == original.bias.item_offsets_)
    assert np.all(algo.user_components_ == original.user_components_)


@need_skl
@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_svd_batch_accuracy():
    from lenskit.algorithms import basic
    from lenskit.algorithms import bias
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.ratings

    svd_algo = svd.BiasedSVD(25, damping=10)
    algo = basic.Fallback(svd_algo, bias.Bias(damping=10))

    def eval(train, test):
        _log.info('running training')
        algo.fit(train)
        _log.info('testing %d users', test.user.nunique())
        return batch.predict(algo, test)

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    preds = pd.concat(eval(train, test) for (train, test) in folds)
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.74, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.92, abs=0.05)
