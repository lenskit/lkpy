import logging

import numpy as np
import pandas as pd

from lenskit import sharing

from pytest import mark

_log = logging.getLogger(__name__)


@mark.parametrize('impl', sharing.share_impls)
def test_share_array_1d(impl):
    with impl() as ctx:
        arr = np.random.randn(5)
        _log.info('saving')
        k = ctx.put_array(arr)
        _log.info('loading')
        a2 = ctx.get_array(k)
        _log.info('testing')
        assert all(a2 == arr)
        _log.info('done')
        del a2


@mark.parametrize('impl', sharing.share_impls)
def test_share_array_2d(impl):
    with impl() as ctx:
        arr = np.random.randn(10, 3)
        k = ctx.put_array(arr)
        a2 = ctx.get_array(k)
        assert np.all(a2 == arr)
        del a2


@mark.parametrize('impl', sharing.share_impls)
def test_share_series(impl):
    with impl() as ctx:
        arr = np.random.randn(100)
        idx = np.random.randint(0, 100000, 100)
        series = pd.Series(arr, index=idx)
        k = ctx.put_series(series)
        a2 = ctx.get_series(k)
        assert all(a2 == arr)
        del a2

@mark.parametrize('impl', sharing.share_impls)
def test_share_index(impl):
    with impl() as ctx:
        idx = np.random.randint(0, 100000, 100)
        index = pd.Index(idx, name='foo')
        k = ctx.put_index(index)
        a2 = ctx.get_index(k)
        assert all(a2.values == idx)
        assert a2.name == 'foo'
        del a2
