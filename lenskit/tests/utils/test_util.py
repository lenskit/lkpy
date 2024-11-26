# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import re
import time

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
    assert s.endswith("ms")


def test_stopwatch_long_str():
    w = lku.Stopwatch()
    time.sleep(1.2)
    s = str(w)
    assert s.endswith("s")


def test_stopwatch_minutes():
    w = lku.Stopwatch()
    w.stop()
    assert w.stop_time is not None
    w.start_time = w.stop_time - 62
    s = str(w)
    p = re.compile(r"1m2.\d\ds")
    assert p.match(s)


def test_stopwatch_hours():
    w = lku.Stopwatch()
    w.stop()
    assert w.stop_time is not None
    w.start_time = w.stop_time - 3663
    s = str(w)
    p = re.compile(r"1h1m3.\d\ds")
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
