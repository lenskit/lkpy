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

from lenskit.data import ItemList

LEFT_SIZE = 50000
RIGHT_SIZE = 100
TOTAL_SIZE = LEFT_SIZE * 2


@mark.benchmark(group="integers")
def test_numpy_integers(rng: np.random.Generator, benchmark):
    left = np.arange(LEFT_SIZE)
    right = rng.choice(LEFT_SIZE * 2, RIGHT_SIZE, replace=False)

    def search():
        _mask = np.isin(left, right)

    benchmark(search)


@mark.benchmark(group="integers")
def test_numpy_integer_mask(rng: np.random.Generator, benchmark):
    left = np.arange(LEFT_SIZE)
    right = rng.choice(LEFT_SIZE * 2, RIGHT_SIZE, replace=False)

    def search():
        mask = np.zeros(TOTAL_SIZE, dtype=np.bool_)
        mask[right] = True
        _m = mask[left]

    benchmark(search)


@mark.benchmark(group="strings")
def test_numpy_strings(rng: np.random.Generator, benchmark):
    left = np.array([str(uuid4()) for _i in range(LEFT_SIZE)])
    right = rng.choice(left, RIGHT_SIZE, replace=False)

    def search():
        _mask = np.isin(left, right)

    benchmark(search)


@mark.benchmark(group="strings")
def test_numpy_dstrings(rng: np.random.Generator, benchmark):
    left = np.array([str(uuid4()) for _i in range(LEFT_SIZE)], dtype=np.dtypes.StringDType)
    right = rng.choice(left, RIGHT_SIZE, replace=False)

    def search():
        _mask = np.isin(left, right)

    benchmark(search)


@mark.benchmark(group="integers")
def test_arrow_integers(rng: np.random.Generator, benchmark):
    left = np.arange(LEFT_SIZE)
    right = rng.choice(LEFT_SIZE * 2, RIGHT_SIZE, replace=False)
    left = pa.array(left)
    right = pa.array(right)

    def search():
        _mask = pc.is_in(left, right)

    benchmark(search)


@mark.benchmark(group="strings")
def test_arrow_strings(rng: np.random.Generator, benchmark):
    left = np.array([str(uuid4()) for _i in range(LEFT_SIZE)])
    right = rng.choice(left, RIGHT_SIZE, replace=False)

    left = pa.array(left)
    right = pa.array(right)

    def search():
        _mask = pc.is_in(left, right)

    benchmark(search)


@mark.benchmark(group="integers")
def test_index_integers(rng: np.random.Generator, benchmark):
    left = np.arange(LEFT_SIZE)
    right = rng.choice(LEFT_SIZE * 2, RIGHT_SIZE, replace=False)

    def search():
        idx = pd.Index(right)
        res = idx.get_indexer_for(left)
        _mask = res >= 0

    benchmark(search)


@mark.benchmark(group="strings")
def test_index_strings(rng: np.random.Generator, benchmark):
    left = np.array([str(uuid4()) for _i in range(LEFT_SIZE)])
    right = rng.choice(left, RIGHT_SIZE, replace=False)

    def search():
        idx = pd.Index(right)
        res = idx.get_indexer_for(left)
        _mask = res >= 0

    benchmark(search)


@mark.benchmark(group="integers")
def test_itemlist_integers(rng: np.random.Generator, benchmark):
    left = np.arange(LEFT_SIZE)
    right = rng.choice(LEFT_SIZE * 2, RIGHT_SIZE, replace=False)

    ill = ItemList(left)
    ilr = ItemList(right)

    def search():
        _mask = ill.isin(ilr)

    benchmark(search)


@mark.benchmark(group="strings")
def test_itemlist_strings(rng: np.random.Generator, benchmark):
    left = np.array([str(uuid4()) for _i in range(LEFT_SIZE)])
    right = rng.choice(left, RIGHT_SIZE, replace=False)

    ill = ItemList(left)
    ilr = ItemList(right)

    def search():
        _mask = ill.isin(ilr)

    benchmark(search)
