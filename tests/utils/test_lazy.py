# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from concurrent.futures import ThreadPoolExecutor

from lenskit.lazy import lazy_thunk, lazy_value


def test_lazy_value():
    lazy = lazy_value("hi")

    assert lazy.get() == "hi"


def test_lazy_thunk():
    lazy = lazy_thunk(lambda: "hi")

    assert lazy.get() == "hi"


def test_lazy_thunk_called_once():
    data = {"count": 0}

    def thunk():
        data["count"] += 1
        return data["count"]

    lazy = lazy_thunk(thunk)

    assert lazy.get() == 1
    assert data["count"] == 1
    assert lazy.get() == 1
    assert data["count"] == 1


def test_lazy_threaded():
    data = {"count": 0}

    def thunk():
        data["count"] += 1
        return data["count"]

    lazy = lazy_thunk(thunk)

    with ThreadPoolExecutor(4) as pool:
        results = pool.map(lambda _i: lazy.get(), range(20))

    assert all(c == 1 for c in results)
