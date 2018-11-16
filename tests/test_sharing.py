import numpy as np
import pandas as pd

from lenskit import sharing

from pytest import mark

impls = [sharing.DiskShareContext]


@mark.parametrize('impl', impls)
def test_share_array_1d(impl):
    with impl() as ctx:
        arr = np.random.randn(5)
        k = ctx.put_array(arr)
        a2 = ctx.get_array(k)
        assert all(a2 == arr)
        del a2


@mark.parametrize('impl', impls)
def test_share_array_2d(impl):
    with impl() as ctx:
        arr = np.random.randn(10, 3)
        k = ctx.put_array(arr)
        a2 = ctx.get_array(k)
        assert np.all(a2 == arr)
        del a2


@mark.parametrize('impl', impls)
def test_share_series(impl):
    with impl() as ctx:
        arr = np.random.randn(100)
        idx = np.random.randint(0, 100000, 100)
        series = pd.Series(arr, index=idx)
        k = ctx.put_series(series)
        a2 = ctx.get_series(k)
        assert all(a2 == arr)
        del a2
