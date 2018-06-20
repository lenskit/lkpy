import math
import numpy as np
import pandas as pd

from pytest import approx, raises

import lenskit.metrics.predict as pm


def test_rmse_one():
    rmse = pm.rmse([1], [1])
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = pm.rmse([1], [2])
    assert rmse == approx(1)

    rmse = pm.rmse([1], [0.5])
    assert rmse == approx(0.5)


def test_rmse_two():
    rmse = pm.rmse([1, 2], [1, 2])
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = pm.rmse([1, 1], [2, 2])
    assert rmse == approx(1)

    rmse = pm.rmse([1, 3], [3, 1])
    assert rmse == approx(2)


def test_rmse_array_two():
    rmse = pm.rmse(np.array([1, 2]), np.array([1, 2]))
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = pm.rmse(np.array([1, 1]), np.array([2, 2]))
    assert rmse == approx(1)

    rmse = pm.rmse(np.array([1, 3]), np.array([3, 1]))
    assert rmse == approx(2)


def test_rmse_series_two():
    rmse = pm.rmse(pd.Series([1, 2]), pd.Series([1, 2]))
    assert isinstance(rmse, float)
    assert rmse == approx(0)

    rmse = pm.rmse(pd.Series([1, 1]), pd.Series([2, 2]))
    assert rmse == approx(1)

    rmse = pm.rmse(pd.Series([1, 3]), pd.Series([3, 1]))
    assert rmse == approx(2)


def test_rmse_series_subset_axis():
    rmse = pm.rmse(pd.Series([1, 3], ['a', 'c']), pd.Series([3, 4, 1], ['a', 'b', 'c']))
    assert rmse == approx(2)


def test_rmse_series_missing_value_error():
    with raises(ValueError):
        pm.rmse(pd.Series([1, 3], ['a', 'd']), pd.Series([3, 4, 1], ['a', 'b', 'c']))


def test_rmse_series_missing_value_ignore():
    rmse = pm.rmse(pd.Series([1, 3], ['a', 'd']), pd.Series([3, 4, 1], ['a', 'b', 'c']),
                   missing='ignore')
    assert rmse == approx(2)
