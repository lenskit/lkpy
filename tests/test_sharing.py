import logging

import numpy as np
import pandas as pd

from lenskit import sharing

_log = logging.getLogger(__name__)


def test_share_array_1d():
    arr = np.random.randn(5)
    _log.info('saving')
    k = sharing.put_array(arr)
    _log.info('loading')
    a2 = sharing.get_array(k)
    _log.info('testing')
    assert all(a2 == arr)
    _log.info('done')
    del a2


def test_share_array_2d():
    arr = np.random.randn(10, 3)
    k = sharing.put_array(arr)
    a2 = sharing.get_array(k)
    assert np.all(a2 == arr)
    del a2


def test_share_series():
    arr = np.random.randn(100)
    idx = np.random.randint(0, 100000, 100)
    series = pd.Series(arr, index=idx)
    k = sharing.put_series(series)
    a2 = sharing.get_series(k)
    assert all(a2 == arr)
    del a2


def test_share_index():
    idx = np.random.randint(0, 100000, 100)
    index = pd.Index(idx, name='foo')
    k = sharing.put_index(index)
    a2 = sharing.get_index(k)
    assert all(a2.values == idx)
    assert a2.name == 'foo'
    del a2
