import logging
from pathlib import Path

import lenskit.algorithms.funksvd as svd

import pandas as pd
import numpy as np

from pytest import approx, mark

import lk_test_utils as lktu

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


def test_fsvd_basic_build():
    algo = svd.FunkSVD(20, iterations=20)
    algo.fit(simple_df)

    assert algo.global_bias_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)


def test_fsvd_clamp_build():
    algo = svd.FunkSVD(20, iterations=20, range=(1, 5))
    algo.fit(simple_df)

    assert algo.global_bias_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)


def test_fsvd_predict_basic():
    algo = svd.FunkSVD(20, iterations=20)
    algo.fit(simple_df)

    assert algo.global_bias_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3] <= 5


def test_fsvd_predict_clamp():
    algo = svd.FunkSVD(20, iterations=20, range=(1, 5))
    algo.fit(simple_df)

    assert algo.global_bias_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert preds.loc[3] >= 1
    assert preds.loc[3] <= 5


def test_fsvd_no_bias():
    algo = svd.FunkSVD(20, iterations=20, bias=None)
    algo.fit(simple_df)

    assert algo.global_bias_ == 0
    assert algo.item_bias_ is None
    assert algo.user_bias_ is None
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert all(preds.notna())


def test_fsvd_predict_bad_item():
    algo = svd.FunkSVD(20, iterations=20)
    algo.fit(simple_df)

    assert algo.global_bias_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_fsvd_predict_bad_item_clamp():
    algo = svd.FunkSVD(20, iterations=20, range=(1, 5))
    algo.fit(simple_df)

    assert algo.global_bias_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(10, [4])
    assert len(preds) == 1
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])


def test_fsvd_predict_bad_user():
    algo = svd.FunkSVD(20, iterations=20)
    algo.fit(simple_df)

    assert algo.global_bias_ == approx(simple_df.rating.mean())
    assert algo.item_features_.shape == (3, 20)
    assert algo.user_features_.shape == (3, 20)

    preds = algo.predict_for_user(50, [3])
    assert len(preds) == 1
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])


@mark.slow
def test_fsvd_save_load(tmp_path):
    tmp_path = lktu.norm_path(tmp_path)
    mod_file = tmp_path / 'funksvd.npz'

    ratings = lktu.ml_pandas.renamed.ratings

    original = svd.FunkSVD(20, iterations=20)
    original.fit(ratings)

    assert original.global_bias_ == approx(ratings.rating.mean())
    assert original.item_features_.shape == (ratings.item.nunique(), 20)
    assert original.user_features_.shape == (ratings.user.nunique(), 20)

    original.save(mod_file)
    assert mod_file.exists()

    algo = svd.FunkSVD(20, iterations=20)
    algo.load(mod_file)
    assert algo.global_bias_ == original.global_bias_
    assert np.all(algo.user_bias_ == original.user_bias_)
    assert np.all(algo.item_bias_ == original.item_bias_)
    assert np.all(algo.user_features_ == original.user_features_)
    assert np.all(algo.item_features_ == original.item_features_)
    assert np.all(algo.item_index_ == original.item_index_)
    assert np.all(algo.user_index_ == original.user_index_)


@mark.slow
def test_fsvd_known_preds():
    algo = svd.FunkSVD(15, iterations=125, lrate=0.001)
    _log.info('training %s on ml data', algo)
    algo.fit(lktu.ml_pandas.renamed.ratings)

    dir = Path(__file__).parent
    pred_file = dir / 'funksvd-preds.csv'
    _log.info('reading known predictions from %s', pred_file)
    known_preds = pd.read_csv(str(pred_file))
    pairs = known_preds.loc[:, ['user', 'item']]

    preds = algo.predict(pairs)
    known_preds.rename(columns={'prediction': 'expected'}, inplace=True)
    merged = known_preds.assign(prediction=preds)
    merged['error'] = merged.expected - merged.prediction
    assert not any(merged.prediction.isna() & merged.expected.notna())
    err = merged.error
    err = err[err.notna()]
    try:
        assert all(err.abs() < 0.01)
    except AssertionError as e:
        bad = merged[merged.error.notna() & (merged.error.abs() >= 0.01)]
        _log.error('erroneous predictions:\n%s', bad)
        raise e


@mark.slow
@mark.eval
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_fsvd_batch_accuracy():
    from lenskit.algorithms import basic
    import lenskit.crossfold as xf
    from lenskit import batch
    import lenskit.metrics.predict as pm

    ratings = lktu.ml100k.load_ratings()

    svd_algo = svd.FunkSVD(25, 125, damping=10)
    algo = basic.Fallback(svd_algo, basic.Bias(damping=10))

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
