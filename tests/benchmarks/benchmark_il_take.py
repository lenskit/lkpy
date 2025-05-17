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
import torch

from pytest import mark

from lenskit.data import ItemList, Vocabulary


@mark.benchmark(group="take")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_itemlist_take(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)

    items = ItemList(np.arange(size), scores=scores)
    ranks = np.argsort(scores)

    def take():
        _il = items[ranks]

    benchmark(take)


@mark.benchmark(group="take")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_arrow_table_take(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)

    items = ItemList(np.arange(size), scores=scores)
    table = items.to_arrow()
    ranks = np.argsort(scores)
    ranks = pa.array(ranks, type=pa.int32())

    def take():
        _il = table.take(ranks)

    benchmark(take)


@mark.benchmark(group="take")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_arrow_struct_take(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)

    items = ItemList(np.arange(size), scores=scores)
    table = items.to_arrow(type="array")
    ranks = np.argsort(scores)
    ranks = pa.array(ranks, type=pa.int32())

    def take():
        _il = table.take(ranks)

    benchmark(take)


@mark.benchmark(group="take")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_arrow_table_take_cvt(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)

    items = ItemList(np.arange(size), scores=scores)
    table = items.to_arrow()
    ranks = np.argsort(scores)

    def take():
        aranks = pa.array(ranks, type=pa.int32())
        _il = table.take(aranks)

    benchmark(take)


@mark.benchmark(group="take")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_pandas_take(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)

    items = ItemList(np.arange(size), scores=scores)
    df = items.to_df()
    ranks = np.argsort(scores)

    def take():
        _il = df.iloc[ranks]

    benchmark(take)


@mark.benchmark(group="take")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_numpy_take(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)

    items = ItemList(np.arange(size), scores=scores)
    ranks = np.argsort(scores)

    def take():
        _ids = items.ids()[ranks]
        _scores = items.ids()[ranks]

    benchmark(take)
