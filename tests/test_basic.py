import os.path

from lenskit.algorithms import basic

import pandas as pd
import numpy as np

import lk_test_utils as lktu
from lk_test_utils import tmpdir
from pytest import approx

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})


def test_precomputed():
    algo = basic.Memorized(simple_df)

    preds = algo.predict(None, 10, [1, 2])
    assert set(preds.index) == set([1, 2])
    assert all(preds == pd.Series({1: 4.0, 2: 5.0}))

    preds = algo.predict(None, 12, [1, 3])
    assert set(preds.index) == set([1, 3])
    assert preds.loc[1] == 3.0
    assert np.isnan(preds.loc[3])


def test_fallback_train_one():
    algo = basic.Fallback(basic.Bias())
    model = algo.train(lktu.ml_pandas.renamed.ratings)
    assert len(model) == 1
    assert isinstance(model[0], basic.BiasModel)
    assert model[0].mean == approx(lktu.ml_pandas.ratings.rating.mean())


def test_fallback_train_one_pred_impossible():
    algo = basic.Fallback(basic.Memorized(simple_df))
    model = algo.train(lktu.ml_pandas.renamed.ratings)
    assert len(model) == 1

    preds = algo.predict(model, 10, [1, 2])
    assert set(preds.index) == set([1, 2])
    assert all(preds == pd.Series({1: 4.0, 2: 5.0}))

    preds = algo.predict(model, 12, [1, 3])
    assert set(preds.index) == set([1, 3])
    assert preds.loc[1] == 3.0
    assert np.isnan(preds.loc[3])


def test_fallback_predict():
    algo = basic.Fallback(basic.Memorized(simple_df), basic.Bias())
    model = algo.train(lktu.ml_pandas.renamed.ratings)
    assert len(model) == 2
    assert isinstance(model[1], basic.BiasModel)
    assert model[1].mean == approx(lktu.ml_pandas.ratings.rating.mean())

    # first user + item
    preds = algo.predict(model, 10, [1])
    assert preds.loc[1] == 4.0
    # second user + first item
    preds = algo.predict(model, 15, [1])
    assert preds.loc[1] == approx(model[1].mean + model[1].users.loc[15] + model[1].items.loc[1])

    # second item + user item
    preds = algo.predict(model, 12, [2])
    assert preds.loc[2] == approx(model[1].mean + model[1].users.loc[12] + model[1].items.loc[2])

    # blended
    preds = algo.predict(model, 10, [1, 5])
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(model[1].mean + model[1].users.loc[10] + model[1].items.loc[5])

    # blended unknown
    preds = algo.predict(model, 10, [5, 1, -23081])
    assert len(preds) == 3
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(model[1].mean + model[1].users.loc[10] + model[1].items.loc[5])
    assert preds.loc[-23081] == approx(model[1].mean + model[1].users.loc[10])


def test_fallback_save_load(tmpdir):
    algo = basic.Fallback(basic.Memorized(simple_df), basic.Bias())
    original = algo.train(lktu.ml_pandas.renamed.ratings)

    fn = os.path.join(tmpdir, 'fallback')
    algo.save_model(original, fn)

    model = algo.load_model(fn)

    assert len(model) == 2
    assert isinstance(model[1], basic.BiasModel)
    assert model[1].mean == approx(lktu.ml_pandas.ratings.rating.mean())

    # first user + item
    preds = algo.predict(model, 10, [1])
    assert preds.loc[1] == 4.0
    # second user + first item
    preds = algo.predict(model, 15, [1])
    assert preds.loc[1] == approx(model[1].mean + model[1].users.loc[15] + model[1].items.loc[1])

    # second item + user item
    preds = algo.predict(model, 12, [2])
    assert preds.loc[2] == approx(model[1].mean + model[1].users.loc[12] + model[1].items.loc[2])

    # blended
    preds = algo.predict(model, 10, [1, 5])
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(model[1].mean + model[1].users.loc[10] + model[1].items.loc[5])

    # blended unknown
    preds = algo.predict(model, 10, [5, 1, -23081])
    assert len(preds) == 3
    assert preds.loc[1] == 4.0
    assert preds.loc[5] == approx(model[1].mean + model[1].users.loc[10] + model[1].items.loc[5])
    assert preds.loc[-23081] == approx(model[1].mean + model[1].users.loc[10])
