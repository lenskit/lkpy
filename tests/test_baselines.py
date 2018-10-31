import lenskit.algorithms.basic as bl

import os.path
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
    model = algo.train(simple_df)
    assert model.mean == approx(3.5)

    assert model.items is not None
    assert model.items.index.name == 'item'
    assert set(model.items.index) == set([1, 2, 3])
    assert model.items.loc[1:3].values == approx(np.array([0, 1.5, -1.5]))

    assert model.users is not None
    assert model.users.index.name == 'user'
    assert set(model.users.index) == set([10, 12, 13])
    assert model.users.loc[[10, 12, 13]].values == approx(np.array([0.25, -0.5, 0]))


def test_bias_global_only():
    algo = bl.Bias(users=False, items=False)
    model = algo.train(simple_df)
    assert model.mean == approx(3.5)
    assert model.items is None
    assert model.users is None


def test_bias_no_user():
    algo = bl.Bias(users=False)
    model = algo.train(simple_df)
    assert model.mean == approx(3.5)

    assert model.items is not None
    assert model.items.index.name == 'item'
    assert set(model.items.index) == set([1, 2, 3])
    assert model.items.loc[1:3].values == approx(np.array([0, 1.5, -1.5]))

    assert model.users is None


def test_bias_no_item():
    algo = bl.Bias(items=False)
    model = algo.train(simple_df)
    assert model.mean == approx(3.5)
    assert model.items is None

    assert model.users is not None
    assert model.users.index.name == 'user'
    assert set(model.users.index) == set([10, 12, 13])
    assert model.users.loc[[10, 12, 13]].values == approx(np.array([1.0, -0.5, -1.5]))


def test_bias_global_predict():
    algo = bl.Bias(items=False, users=False)
    model = algo.train(simple_df)
    p = algo.predict(model, 10, [1, 2, 3])

    assert len(p) == 3
    assert (p == model.mean).all()
    assert p.values == approx(model.mean)


def test_bias_item_predict():
    algo = bl.Bias(users=False)
    model = algo.train(simple_df)
    p = algo.predict(model, 10, [1, 2, 3])

    assert len(p) == 3
    assert p.values == approx((model.items + model.mean).values)


def test_bias_user_predict():
    algo = bl.Bias(items=False)
    model = algo.train(simple_df)
    p = algo.predict(model, 10, [1, 2, 3])

    assert len(p) == 3
    assert p.values == approx(model.mean + 1.0)

    p = algo.predict(model, 12, [1, 3])

    assert len(p) == 2
    assert p.values == approx(model.mean - 0.5)


def test_bias_new_user_predict():
    algo = bl.Bias()
    model = algo.train(simple_df)

    ratings = pd.DataFrame({'item': [1, 2, 3], 'rating': [1.5, 2.5, 3.5]})
    ratings = ratings.set_index('item').rating
    p = algo.predict(model, None, [1, 3], ratings=ratings)

    offs = ratings - model.mean - model.items
    umean = offs.mean()
    _log.info('user mean is %f', umean)

    assert len(p) == 2
    assert p.values == approx((model.mean + model.items + umean).loc[[1, 3]].values)


def test_bias_predict_unknown_item():
    algo = bl.Bias()
    model = algo.train(simple_df)

    p = algo.predict(model, 10, [1, 3, 4])

    assert len(p) == 3
    assert p.loc[[1, 3]].values == approx((model.items.loc[[1, 3]] + model.mean + 0.25).values)
    assert p.loc[4] == approx(model.mean + 0.25)


def test_bias_predict_unknown_user():
    algo = bl.Bias()
    model = algo.train(simple_df)

    p = algo.predict(model, 15, [1, 3])

    assert len(p) == 2
    assert p.values == approx((model.items.loc[[1, 3]] + model.mean).values)


def test_bias_train_ml_ratings():
    algo = bl.Bias()
    ratings = ml_pandas.ratings.rename(columns={'userId': 'user', 'movieId': 'item'})
    model = algo.train(ratings)

    assert model.mean == approx(ratings.rating.mean())
    imeans_data = ratings.groupby('item').rating.mean()
    imeans_algo = model.items + model.mean
    ares, data = imeans_algo.align(imeans_data)
    assert ares.values == approx(data.values)

    urates = ratings.set_index('user').loc[2].set_index('item').rating
    umean = (urates - imeans_data[urates.index]).mean()
    p = algo.predict(model, 2, [10, 11, -1])
    assert len(p) == 3
    assert p.iloc[0] == approx(imeans_data.loc[10] + umean)
    assert p.iloc[1] == approx(imeans_data.loc[11] + umean)
    assert p.iloc[2] == approx(ratings.rating.mean() + umean)


def test_bias_item_damp():
    algo = bl.Bias(users=False, damping=5)
    model = algo.train(simple_df)
    assert model.mean == approx(3.5)

    assert model.items is not None
    assert model.items.index.name == 'item'
    assert set(model.items.index) == set([1, 2, 3])
    assert model.items.loc[1:3].values == approx(np.array([0, 0.25, -0.25]))

    assert model.users is None


def test_bias_user_damp():
    algo = bl.Bias(items=False, damping=5)
    model = algo.train(simple_df)
    assert model.mean == approx(3.5)
    assert model.items is None

    assert model.users is not None
    assert model.users.index.name == 'user'
    assert set(model.users.index) == set([10, 12, 13])
    assert model.users.loc[[10, 12, 13]].values == \
        approx(np.array([0.2857, -0.08333, -0.25]), abs=1.0e-4)


def test_bias_damped():
    algo = bl.Bias(damping=5)
    model = algo.train(simple_df)
    assert model.mean == approx(3.5)

    assert model.items is not None
    assert model.items.index.name == 'item'
    assert set(model.items.index) == set([1, 2, 3])
    assert model.items.loc[1:3].values == approx(np.array([0, 0.25, -0.25]))

    assert model.users is not None
    assert model.users.index.name == 'user'
    assert set(model.users.index) == set([10, 12, 13])
    assert model.users.loc[[10, 12, 13]].values == \
        approx(np.array([0.25, -00.08333, -0.20833]), abs=1.0e-4)


def test_bias_separate_damping():
    algo = bl.Bias(damping=(5, 10))
    model = algo.train(simple_df)
    assert model.mean == approx(3.5)

    assert model.items is not None
    assert model.items.index.name == 'item'
    assert set(model.items.index) == set([1, 2, 3])
    assert model.items.loc[1:3].values == \
        approx(np.array([0, 0.136364, -0.13636]), abs=1.0e-4)

    assert model.users is not None
    assert model.users.index.name == 'user'
    assert set(model.users.index) == set([10, 12, 13])
    assert model.users.loc[[10, 12, 13]].values == \
        approx(np.array([0.266234, -0.08333, -0.22727]), abs=1.0e-4)


def test_bias_save(tmp_path):
    tmp_path = lktu.norm_path(tmp_path)

    algo = bl.Bias(damping=5)
    original = algo.train(simple_df)
    assert original.mean == approx(3.5)
    fn = tmp_path / 'bias.dat'

    _log.info('saving to %s', fn)
    algo.save_model(original, fn)

    model = algo.load_model(fn)
    assert model is not original
    assert model.mean == original.mean

    assert model.items is not None
    assert model.items.index.name == 'item'
    assert set(model.items.index) == set([1, 2, 3])
    assert model.items.loc[1:3].values == approx(np.array([0, 0.25, -0.25]))

    assert model.users is not None
    assert model.users.index.name == 'user'
    assert set(model.users.index) == set([10, 12, 13])
    assert model.users.loc[[10, 12, 13]].values == \
        approx(np.array([0.25, -00.08333, -0.20833]), abs=1.0e-4)
