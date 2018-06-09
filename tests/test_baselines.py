import lenskit.algorithms.baselines as bl

import pandas as pd
import numpy as np

from pytest import approx

def test_bias_basic_build():
    df = pd.DataFrame({'item': [1,1,2,3], 'user': [10,12,10,13], 'rating': [4.0,3.0,5.0,2.0]})
    algo = bl.Bias()
    model = algo.train(df)
    assert model.mean == approx(3.5)

    assert model.items is not None
    assert set(model.items.index) == set([1,2,3])
    assert model.items.loc[1:3].values == approx(np.array([0, 1.5, -1.5]))
    
    assert model.users is not None
    assert set(model.users.index) == set([10, 12, 13])
    assert model.users.loc[[10,12,13]].values == approx(np.array([0.25,-0.5,0]))

def test_bias_global_only():
    df = pd.DataFrame({'item': [1,1,2,3], 'user': [10,12,10,13], 'rating': [4.0,3.0,5.0,2.0]})
    algo = bl.Bias(users=False, items=False)
    model = algo.train(df)
    assert model.mean == approx(3.5)
    assert model.items is None
    assert model.users is None

def test_bias_no_user():
    df = pd.DataFrame({'item': [1,1,2,3], 'user': [10,12,10,13], 'rating': [4.0,3.0,5.0,2.0]})
    algo = bl.Bias(users=False)
    model = algo.train(df)
    assert model.mean == approx(3.5)
    
    assert model.items is not None
    assert set(model.items.index) == set([1,2,3])
    assert model.items.loc[1:3].values == approx(np.array([0, 1.5, -1.5]))
    
    assert model.users is None

def test_bias_no_item():
    df = pd.DataFrame({'item': [1,1,2,3], 'user': [10,12,10,13], 'rating': [4.0,3.0,5.0,2.0]})
    algo = bl.Bias(items=False)
    model = algo.train(df)
    assert model.mean == approx(3.5)
    assert model.items is None

    assert model.users is not None
    assert set(model.users.index) == set([10, 12, 13])
    assert model.users.loc[[10,12,13]].values == approx(np.array([1.0, -0.5, -1.5]))