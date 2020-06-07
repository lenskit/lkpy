import logging
from pytest import approx, mark, skip, fixture

import pandas as pd
import numpy as np
import binpickle

import lenskit.util.test as lktu
from lenskit.algorithms import Recommender

try:
    from lenskit.algorithms import tf as lktf
    import tensorflow as tf
except ImportError:
    pytestmark = mark.skip('tensorflow not available')

_log = logging.getLogger(__name__)


@fixture(scope='function')
def tf_session():
    tf.keras.backend.clear_session()


@mark.slow
def test_tf_bias_save_load(tmp_path, tf_session):
    "Training, saving, and loading a bias model."
    fn = tmp_path / 'bias.bpk'
    ratings = lktu.ml_test.ratings

    original = lktf.BiasedMF(20, batch_size=1024, epochs=20)
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

    preds = algo.predict_for_user(100, [5, 10, 30])
    assert all(preds.notna())


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_tf_bias_batch_accuracy(tf_session):
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.ratings

    algo = lktf.BiasedMF(25, damping=10, batch_size=1024, epochs=20, rng_spec=42)
    algo = basic.Fallback(algo, basic.Bias(damping=10))

    def eval(train, test):
        _log.info('running training')
        algo.fit(train)
        _log.info('testing %d users', test.user.nunique())
        return batch.predict(algo, test)

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    preds = pd.concat(eval(train, test) for (train, test) in folds)
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.83, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(1.03, abs=0.05)


@mark.slow
def test_tf_ibias_general(tmp_path, tf_session):
    "Training, saving, loading, and using an integrated bias model."
    fn = tmp_path / 'bias.bpk'
    ratings = lktu.ml_test.ratings

    original = lktf.IntegratedBiasMF(20, batch_size=1024, epochs=20, rng_spec=42)
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
def test_tf_ibias_batch_accuracy(n_jobs, tf_session):
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.ratings

    algo = lktf.IntegratedBiasMF(20, batch_size=1024, epochs=20, rng_spec=42)
    algo = basic.Fallback(algo, basic.Bias(damping=10))

    def eval(train, test):
        _log.info('running training')
        algo.fit(train)
        _log.info('testing %d users', test.user.nunique())
        return batch.predict(algo, test, n_jobs=n_jobs)

    folds = xf.partition_users(ratings, 5, xf.SampleFrac(0.2))
    preds = pd.concat(eval(train, test) for (train, test) in folds)
    mae = pm.mae(preds.prediction, preds.rating)
    assert mae == approx(0.73, abs=0.025)

    user_rmse = preds.groupby('user').apply(lambda df: pm.rmse(df.prediction, df.rating))
    assert user_rmse.mean() == approx(0.93, abs=0.05)


@mark.slow
def test_tf_bpr_general(tmp_path, tf_session):
    "Training, saving, loading, and using a BPR model."
    fn = tmp_path / 'bias.bpk'
    ratings = lktu.ml_test.ratings

    original = lktf.BPR(20, batch_size=1024, epochs=20, neg_count=2, rng_spec=42)
    original.fit(ratings)
    with original.graph.as_default():
        ue = original.model.get_layer('user-embed')
        assert ue.get_weights()[0].shape == (ratings.user.nunique(), 20)
        ie = original.model.get_layer('item-embed')
        assert ie.get_weights()[0].shape == (ratings.item.nunique(), 20)

    binpickle.dump(original, fn)

    _log.info('serialized to %d bytes', fn.stat().st_size)
    algo = binpickle.load(fn)

    # does scoring work?
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
def test_tf_bpr_batch_accuracy(tf_session):
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch, topn

    ratings = lktu.ml100k.ratings

    algo = lktf.BPR(20, batch_size=1024, epochs=20, rng_spec=42)
    algo = Recommender.adapt(algo)

    all_recs = []
    all_test = []
    for train, test in xf.partition_users(ratings, 5, xf.SampleFrac(0.2)):
        _log.info('running training')
        algo.fit(train)
        _log.info('testing %d users', test.user.nunique())
        recs = batch.recommend(algo, np.unique(test.user), 50)
        all_recs.append(recs)
        all_test.append(test)

    _log.info('analyzing results')
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    rla.add_metric(topn.recip_rank)
    scores = rla.compute(pd.concat(all_recs, ignore_index=True),
                         pd.concat(all_test, ignore_index=True),
                         include_missing=True)
    scores.fillna(0, inplace=True)
    _log.info('MRR: %f', scores['recip_rank'].mean())
    _log.info('nDCG: %f', scores['ndcg'].mean())
    assert scores['ndcg'].mean() > 0.1
