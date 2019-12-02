import time
import re
import pathlib

import numpy as np
import pandas as pd

from lenskit import util as lku


def test_stopwatch_instant():
    w = lku.Stopwatch()
    assert w.elapsed() > 0


def test_stopwatch_sleep():
    w = lku.Stopwatch()
    time.sleep(0.5)
    assert w.elapsed() >= 0.45


def test_stopwatch_stop():
    w = lku.Stopwatch()
    time.sleep(0.5)
    w.stop()
    time.sleep(0.5)
    assert w.elapsed() >= 0.45


def test_stopwatch_str():
    w = lku.Stopwatch()
    time.sleep(0.5)
    s = str(w)
    assert s.endswith('ms')


def test_stopwatch_long_str():
    w = lku.Stopwatch()
    time.sleep(1.2)
    s = str(w)
    assert s.endswith('s')


def test_stopwatch_minutes():
    w = lku.Stopwatch()
    w.stop()
    w.start_time = w.stop_time - 62
    s = str(w)
    p = re.compile(r'1m2.\d\ds')
    assert p.match(s)


def test_stopwatch_hours():
    w = lku.Stopwatch()
    w.stop()
    w.start_time = w.stop_time - 3663
    s = str(w)
    p = re.compile(r'1h1m3.\d\ds')
    assert p.match(s)


def test_last_memo():
    history = []

    def func(foo):
        history.append(foo)
    cache = lku.LastMemo(func)

    cache("foo")
    assert len(history) == 1
    # string literals are interned
    cache("foo")
    assert len(history) == 1
    cache("bar")
    assert len(history) == 2


def test_write_parquet(tmp_path):
    assert tmp_path.exists()
    fn = tmp_path / 'out.parquet'
    frame = pd.DataFrame({'n': np.arange(10), 'x': np.random.randn(10) + 5})
    lku.write_parquet(fn, frame)

    f2 = pd.read_parquet(fn)
    assert all(f2.n == frame.n)
    assert all(f2.x == frame.x)


def test_append_parquet(tmp_path):
    fn = tmp_path / 'out.parquet'
    frame = pd.DataFrame({'n': np.arange(10), 'x': np.random.randn(10) + 5})
    lku.write_parquet(fn, frame.iloc[:5], True)
    lku.write_parquet(fn, frame.iloc[5:], True)

    f2 = pd.read_parquet(fn)
    assert all(f2.n == frame.n)
    assert all(f2.x == frame.x)


def test_read_ml():
    ratings = lku.load_ml_ratings()
    assert len(ratings) > 100000
    assert set(ratings.columns) == set(['user', 'item', 'rating', 'timestamp'])
