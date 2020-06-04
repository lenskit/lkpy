import logging
from pytest import approx, mark, skip

import pandas as pd
import numpy as np
import binpickle

import lenskit.util.test as lktu

try:
    from lenskit.algorithms import tf as lktf
except ImportError:
    skip('tensorflow not available')

_log = logging.getLogger(__name__)


@mark.slow
def test_tf_bias_save_load(tmp_path):
    "Training, saving, and loading a bias model."
    fn = tmp_path / 'bias.bpk'
    ratings = lktu.ml_test.ratings

    original = lktf.BiasedMF(20, batch_size=1024)
    original.fit(ratings)
    assert original.user_features_.shape == (ratings.user.nunique(), 20)
    assert original.item_features_.shape == (ratings.item.nunique(), 20)

    binpickle.dump(original, fn)

    _log.info('serialized to %d bytes', fn.stat().st_size)
    algo = binpickle.load(fn)

    assert algo.bias.mean_ == original.bias.mean_
    assert np.all(algo.bias.user_offsets_ == original.bias.user_offsets_)
    assert np.all(algo.bias.item_offsets_ == original.bias.item_offsets_)
    assert np.all(algo.user_features_ == original.user_features_)
    assert np.all(algo.item_features_ == original.item_features_)

    preds = algo.predict(100, [5, 10, 30])
    assert all(preds.notna())


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_tf_bias_batch_accuracy():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.ratings

    algo = lktf.BiasedMF(25, damping=10, batch_size=1024)
    algo = basic.Fallback(algo, basic.Bias(damping=10))

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


@mark.slow
def test_tf_ibias_general(tmp_path):
    "Training, saving, loading, and using an integrated bias model."
    fn = tmp_path / 'bias.bpk'
    ratings = lktu.ml_test.ratings

    original = lktf.IntegratedBiasMF(20, batch_size=1024, epochs=50)
    original.fit(ratings)
    with original.graph.as_default():
        ue = original.model.get_layer('user-embed')
        assert ue.get_weights()[0].shape == (ratings.user.nunique(), 20)
        ie = original.model.get_layer('item-embed')
        assert ie.get_weights()[0].shape == (ratings.item.nunique(), 20)

    binpickle.dump(original, fn)

    _log.info('serialized to %d bytes', fn.stat().st_size)
    algo = binpickle.load(fn)

    # does predicting work?
    preds = algo.predict_for_user(100, [5, 10, 30])
    assert all(preds.notna())

    # can we include a nonexistent item?
    preds = algo.predict_for_user(100, [5, 10, 230413804])
    assert len(preds) == 3
    assert all(preds.loc[[230413804]].isna())
    assert preds.isna().sum() == 1


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
@mark.parametrize('n_jobs', [1, None])
def test_tf_ibias_batch_accuracy(n_jobs):
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.ratings

    algo = lktf.IntegratedBiasMF(20, batch_size=1024, epochs=50)
    algo = basic.Fallback(algo, basic.Bias(damping=10))

    def eval(train, test):
        _log.info('running training')
        algo.fit(train)
        _log.info('testing %d users', test.user.nunique())
        return batch.predict(algo, test, n_jobs=n_jobs)

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    preds = pd.concat(eval(train, test) for (train, test) in folds)
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.74, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.92, abs=0.05)
