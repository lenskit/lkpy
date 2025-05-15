# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from uuid import uuid4

import numexpr
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from pytest import mark

from lenskit.data import ItemList, Vocabulary

LEFT_SIZE = 50000
RIGHT_SIZE = 100
TOTAL_SIZE = LEFT_SIZE * 2


@mark.benchmark()
@mark.parametrize("size", [100, 1000, 50000])
def test_numpy(benchmark, size: int):
    def search():
        ranks = np.arange(1, size + 1)
        _w = np.power(0.85, ranks - 1)

    benchmark(search)


@mark.benchmark()
@mark.parametrize("size", [100, 1000, 50000])
def test_numexpr(benchmark, size: int):
    def search():
        ranks = np.arange(1, size + 1)
        _w = numexpr.evaluate("patience ** (rank - 1)", {"patience": 0.85, "rank": ranks})

    benchmark(search)
