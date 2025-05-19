# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from pytest import mark

from lenskit.data import ItemList, Vocabulary

LEFT_SIZE = 50000
RIGHT_SIZE = 100
TOTAL_SIZE = LEFT_SIZE * 2


@mark.benchmark(group="numbers")
@mark.parametrize("size", [100, 5000, 10_000, 100_000, 1_000_000])
def test_itemlist_remove_numbers(rng: np.random.Generator, size: int, benchmark):
    vocab = Vocabulary(np.arange(size * 2))
    left = np.arange(size)
    right = rng.choice(size * 2, RIGHT_SIZE, replace=False)

    il_left = ItemList(item_nums=left, vocabulary=vocab, scores=rng.standard_exponential(size))

    def search():
        _rem = il_left.remove(numbers=right)

    benchmark(search)


@mark.benchmark(group="numbers")
@mark.parametrize("size", [100, 5000, 10_000, 100_000, 1_000_000])
def test_numpy_isin_numbers(rng: np.random.Generator, size: int, benchmark):
    vocab = Vocabulary(np.arange(size * 2))
    left = np.arange(size)
    right = rng.choice(size * 2, RIGHT_SIZE, replace=False)

    il_left = ItemList(item_nums=left, vocabulary=vocab, scores=rng.standard_exponential(size))

    def search():
        mask = np.isin(il_left.numbers(), right)
        _rem = il_left[mask]

    benchmark(search)
