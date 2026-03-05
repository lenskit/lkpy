# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from pytest import importorskip, mark

from lenskit.data import ItemList, Vocabulary

numexpr = importorskip("numexpr")

LEFT_SIZE = 50000
RIGHT_SIZE = 100
TOTAL_SIZE = LEFT_SIZE * 2


@mark.benchmark(group="geometric")
@mark.parametrize("size", [100, 1000, 50000])
def test_numpy_geom(benchmark, size: int):
    ranks = np.arange(1, size + 1)

    def search():
        _w = np.power(0.85, ranks - 1)

    benchmark(search)


@mark.benchmark(group="geometric")
@mark.parametrize("size", [100, 1000, 50000])
def test_numpy_geom_log(benchmark, size: int):
    ranks = np.arange(1, size + 1)

    def search():
        _w = np.exp(np.log(0.85) * (ranks - 1))

    benchmark(search)


@mark.benchmark(group="geometric")
@mark.parametrize("size", [100, 1000, 50000])
def test_numpy_geom_log_ip(benchmark, size: int):
    ranks = np.arange(1, size + 1)

    def search():
        scores = np.require(ranks, dtype=np.float64)
        scores -= 1
        scores *= np.log(0.85)
        np.exp(scores, out=scores)

    benchmark(search)


@mark.benchmark(group="geometric")
@mark.parametrize("size", [100, 1000, 50000])
def test_numexpr_geom(benchmark, size: int):
    ranks = np.arange(1, size + 1)

    def search():
        _w = numexpr.evaluate("patience ** (rank - 1)", {"patience": 0.85, "rank": ranks})

    benchmark(search)


@mark.benchmark(group="geometric")
@mark.parametrize("size", [100, 1000, 50000])
def test_numexpr_geom_log(benchmark, size: int):
    ranks = np.arange(1, size + 1)

    def search():
        _w = numexpr.evaluate("exp(log(patience) * (rank - 1))", {"patience": 0.85, "rank": ranks})

    benchmark(search)


@mark.benchmark(group="logarithmic")
@mark.parametrize("size", [100, 1000, 50000])
def test_numpy_log_old(benchmark, size: int):
    ranks = np.arange(1, size + 1)

    def search():
        _w = np.reciprocal(np.log(np.maximum(ranks, 2)) / np.log(2))

    benchmark(search)


@mark.benchmark(group="logarithmic")
@mark.parametrize("size", [100, 1000, 50000])
def test_numpy_log(benchmark, size: int):
    ranks = np.arange(1, size + 1)

    def search():
        _w = np.log(2) / np.log(np.maximum(ranks, 2))

    benchmark(search)


@mark.benchmark(group="logarithmic")
@mark.parametrize("size", [100, 1000, 50000])
def test_numexpr_log(benchmark, size: int):
    ranks = np.arange(1, size + 1)

    def search():
        _w = numexpr.evaluate(
            "log(base) / log(where(rank < 2, 2, rank))", {"base": 2, "rank": ranks}
        )

    benchmark(search)
