from lenskit.algorithms.basic import Bias
from lenskit import util as lku

import logging
import pickle
import binpickle

import pandas as pd
import numpy as np

from pytest import approx, raises

from lenskit.util.test import ml_test

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


def test_bias_check_arguments():
    # negative damping is not allowed
    with raises(ValueError):
        Bias(damping=-1)

    # negative user damping not allowed
    with raises(ValueError):
        Bias(damping=(-1, 5))

    # negative user damping not allowed
    with raises(ValueError):
        Bias(damping=(5, -1))


def test_bias_full():
    algo = Bias()
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


def test_bias_clone():
    algo = Bias()
    algo.fit(simple_df)

    params = algo.get_params()
    assert sorted(params.keys()) == ['damping', 'items', 'users']

    a2 = lku.clone(algo)
    assert a2 is not algo
    assert getattr(a2, 'mean_', None) is None
    assert getattr(a2, 'item_offsets_', None) is None
    assert getattr(a2, 'user_offsets_', None) is None


def test_bias_global_only():
    algo = Bias(users=False, items=False)
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)
    assert algo.item_offsets_ is None
    assert algo.user_offsets_ is None


def test_bias_no_user():
    algo = Bias(users=False)
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == 'item'
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(np.array([0, 1.5, -1.5]))

    assert algo.user_offsets_ is None


def test_bias_no_item():
    algo = Bias(items=False)
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)
    assert algo.item_offsets_ is None

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == 'user'
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == approx(np.array([1.0, -0.5, -1.5]))


def test_bias_global_predict():
    algo = Bias(items=False, users=False)
    algo.fit(simple_df)
    p = algo.predict_for_user(10, [1, 2, 3])

    assert len(p) == 3
    assert (p == algo.mean_).all()
    assert p.values == approx(algo.mean_)


def test_bias_item_predict():
    algo = Bias(users=False)
    algo.fit(simple_df)
    p = algo.predict_for_user(10, [1, 2, 3])

    assert len(p) == 3
    assert p.values == approx((algo.item_offsets_ + algo.mean_).values)


def test_bias_user_predict():
    algo = Bias(items=False)
    algo.fit(simple_df)
    p = algo.predict_for_user(10, [1, 2, 3])

    assert len(p) == 3
    assert p.values == approx(algo.mean_ + 1.0)

    p = algo.predict_for_user(12, [1, 3])

    assert len(p) == 2
    assert p.values == approx(algo.mean_ - 0.5)


def test_bias_new_user_predict():
    algo = Bias()
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
    algo = Bias()
    algo.fit(simple_df)

    p = algo.predict_for_user(10, [1, 3, 4])

    assert len(p) == 3
    intended = algo.item_offsets_.loc[[1, 3]] + algo.mean_ + 0.25
    assert p.loc[[1, 3]].values == approx(intended.values)
    assert p.loc[4] == approx(algo.mean_ + 0.25)


def test_bias_predict_unknown_user():
    algo = Bias()
    algo.fit(simple_df)

    p = algo.predict_for_user(15, [1, 3])

    assert len(p) == 2
    assert p.values == approx((algo.item_offsets_.loc[[1, 3]] + algo.mean_).values)


def test_bias_train_ml_ratings():
    algo = Bias()
    ratings = ml_test.ratings
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


def test_bias_transform():
    algo = Bias()
    ratings = ml_test.ratings

    normed = algo.fit_transform(ratings)

    assert all(normed['user'] == ratings['user'])
    assert all(normed['item'] == ratings['item'])
    denorm = algo.inverse_transform(normed)
    assert approx(denorm['rating'] == ratings['rating'], 1.0e-6)


def test_bias_item_damp():
    algo = Bias(users=False, damping=5)
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)

    assert algo.item_offsets_ is not None
    assert algo.item_offsets_.index.name == 'item'
    assert set(algo.item_offsets_.index) == set([1, 2, 3])
    assert algo.item_offsets_.loc[1:3].values == approx(np.array([0, 0.25, -0.25]))

    assert algo.user_offsets_ is None


def test_bias_user_damp():
    algo = Bias(items=False, damping=5)
    algo.fit(simple_df)
    assert algo.mean_ == approx(3.5)
    assert algo.item_offsets_ is None

    assert algo.user_offsets_ is not None
    assert algo.user_offsets_.index.name == 'user'
    assert set(algo.user_offsets_.index) == set([10, 12, 13])
    assert algo.user_offsets_.loc[[10, 12, 13]].values == \
        approx(np.array([0.2857, -0.08333, -0.25]), abs=1.0e-4)


def test_bias_damped():
    algo = Bias(damping=5)
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
    algo = Bias(damping=(5, 10))
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


def test_bias_save():
    original = Bias(damping=5)
    original.fit(simple_df)
    assert original.mean_ == approx(3.5)

    _log.info('saving baseline model')
    mod = pickle.dumps(original)
    _log.info('serialized to %d bytes', len(mod))

    algo = pickle.loads(mod)

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


def test_bias_binpickle(tmp_path):
    original = Bias(damping=5)
    original.fit(simple_df)
    assert original.mean_ == approx(3.5)

    _log.info('saving baseline model')
    fn = tmp_path / 'bias.bpk'
    binpickle.dump(original, fn)
    algo = binpickle.load(fn)

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
