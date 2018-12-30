import lenskit.algorithms.basic as bl

import logging

import pandas as pd
import numpy as np

from pytest import approx

import lk_test_utils as lktu
from lk_test_utils import ml_pandas

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


def test_bias_full():
    algo = bl.Bias()
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == 'item'
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(np.array([0, 1.5, -1.5]))

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == 'user'
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == approx(np.array([0.25, -0.5, 0]))


def test_bias_global_only():
    algo = bl.Bias(users=False, items=False)
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)
    assert algo.item_offsets_ is None
    assert algo.user_offsets_ is None


def test_bias_no_user():
    algo = bl.Bias(users=False)
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == 'item'
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(np.array([0, 1.5, -1.5]))

    assert algo.user_offsets_ is None


def test_bias_no_item():
    algo = bl.Bias(items=False)
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)
    assert algo.item_offsets_ is None

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == 'user'
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == approx(np.array([1.0, -0.5, -1.5]))


def test_bias_global_predict():
    algo = bl.Bias(items=False, users=False)
    algo.fit(simple_df)
    p = algo.predict_for_user(10, [1, 2, 3])

    assert len(p) == 3
    assert (p == algo.mean_).all()
    assert p.values == approx(algo.mean_)


def test_bias_item_predict():
    algo = bl.Bias(users=False)
    algo.fit(simple_df)
    p = algo.predict_for_user(10, [1, 2, 3])

    assert len(p) == 3
    assert p.values == approx((algo.item_offsets_ + algo.mean_).values)


def test_bias_user_predict():
    algo = bl.Bias(items=False)
    algo.fit(simple_df)
    p = algo.predict_for_user(10, [1, 2, 3])

    assert len(p) == 3
    assert p.values == approx(algo.mean_ + 1.0)

    p = algo.predict_for_user(12, [1, 3])

    assert len(p) == 2
    assert p.values == approx(algo.mean_ - 0.5)


def test_bias_new_user_predict():
    algo = bl.Bias()
    algo.fit(simple_df)

    ratings = pd.DataFrame({'item': [1, 2, 3], 'rating': [1.5, 2.5, 3.5]})
    ratings = ratings.set_index('item').rating
    p = algo.predict_for_user(None, [1, 3], ratings=ratings)

    offs = ratings - algo.mean_ - algo.item_offsets_
    umean = offs.mean()
    _log.info('user mean is %f', umean)

    assert len(p) == 2
    assert p.values == approx((algo.mean_ + algo.item_offsets_ + umean).loc[[1, 3]].values)


def test_bias_predict_unknown_item():
    algo = bl.Bias()
    algo.fit(simple_df)

    p = algo.predict_for_user(10, [1, 3, 4])

    assert len(p) == 3
    intended = algo.item_offsets_.loc[[1, 3]] + algo.mean_ + 0.25
    assert p.loc[[1, 3]].values == approx(intended.values)
    assert p.loc[4] == approx(algo.mean_ + 0.25)


def test_bias_predict_unknown_user():
    algo = bl.Bias()
    algo.fit(simple_df)

    p = algo.predict_for_user(15, [1, 3])

    assert len(p) == 2
    assert p.values == approx((algo.item_offsets_.loc[[1, 3]] + algo.mean_).values)


def test_bias_train_ml_ratings():
    algo = bl.Bias()
    ratings = ml_pandas.ratings.rename(columns={'userId': 'user', 'movieId': 'item'})
    algo.fit(ratings)

    assert algo.mean_ == approx(ratings.rating.mean())
    imeans_data = ratings.groupby('item').rating.mean()
    imeans_algo = algo.item_offsets_ + algo.mean_
    ares, data = imeans_algo.align(imeans_data)
    assert ares.values == approx(data.values)

    urates = ratings.set_index('user').loc[2].set_index('item').rating
    umean = (urates - imeans_data[urates.index]).mean()
    p = algo.predict_for_user(2, [10, 11, -1])
    assert len(p) == 3
    assert p.iloc[0] == approx(imeans_data.loc[10] + umean)
    assert p.iloc[1] == approx(imeans_data.loc[11] + umean)
    assert p.iloc[2] == approx(ratings.rating.mean() + umean)


def test_bias_item_damp():
    algo = bl.Bias(users=False, damping=5)
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == 'item'
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(np.array([0, 0.25, -0.25]))

    assert algo.user_offsets_ is None


def test_bias_user_damp():
    algo = bl.Bias(items=False, damping=5)
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)
    assert algo.item_offsets_ is None

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == 'user'
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == \
        approx(np.array([0.2857, -0.08333, -0.25]), abs=1.0e-4)


def test_bias_damped():
    algo = bl.Bias(damping=5)
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == 'item'
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(np.array([0, 0.25, -0.25]))

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == 'user'
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == \
        approx(np.array([0.25, -00.08333, -0.20833]), abs=1.0e-4)


def test_bias_separate_damping():
    algo = bl.Bias(damping=(5, 10))
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == 'item'
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == \
        approx(np.array([0, 0.136364, -0.13636]), abs=1.0e-4)

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == 'user'
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == \
        approx(np.array([0.266234, -0.08333, -0.22727]), abs=1.0e-4)


def test_bias_save(tmp_path):
    tmp_path = lktu.norm_path(tmp_path)

    original = bl.Bias(damping=5)
    original.fit(simple_df)
    assert original.mean_ == approx(3.5)
    fn = tmp_path / 'bias.dat'

    _log.info('saving to %s', fn)
    original.save(fn)

    algo = bl.Bias()
    algo.load(fn)
    assert algo.mean_ == original.mean_

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == 'item'
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(np.array([0, 0.25, -0.25]))

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == 'user'
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == \
        approx(np.array([0.25, -00.08333, -0.20833]), abs=1.0e-4)
