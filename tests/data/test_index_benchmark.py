# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from pytest import mark

VOCAB_SIZE = 50000


@mark.benchmark(group="single")
def test_pandas_integer_single(rng: np.random.Generator, benchmark):
    ids = rng.choice(VOCAB_SIZE * 1000, VOCAB_SIZE, replace=False)
    index = pd.Index(ids)

    query = rng.choice(ids, 1).item()

    def search():
        _val = index.get_loc(query)

    benchmark(search)


@mark.benchmark(group="array")
def test_pandas_integer_500(rng: np.random.Generator, benchmark):
    ids = rng.choice(VOCAB_SIZE * 1000, VOCAB_SIZE, replace=False)
    index = pd.Index(ids)

    query = rng.choice(ids, 500, replace=False)

    def search():
        _val = index.get_indexer_for(query)

    benchmark(search)


@mark.benchmark(group="single")
def test_pandas_string_single(rng: np.random.Generator, benchmark):
    ids = [str(uuid4()) for i in range(VOCAB_SIZE)]
    index = pd.Index(ids)

    query = rng.choice(ids, 1).item()

    def search():
        _val = index.get_loc(query)

    benchmark(search)


@mark.benchmark(group="array")
def test_pandas_string_500(rng: np.random.Generator, benchmark):
    ids = [str(uuid4()) for i in range(VOCAB_SIZE)]
    index = pd.Index(ids)

    query = rng.choice(ids, 500, replace=False)

    def search():
        _val = index.get_indexer_for(query)

    benchmark(search)


@mark.benchmark(group="single")
def test_arrow_integer_single(rng: np.random.Generator, benchmark):
    ids = rng.choice(VOCAB_SIZE * 1000, VOCAB_SIZE, replace=False)
    index = pa.array(ids)

    query = rng.choice(ids, 1).item()

    def search():
        _val = index.index(query)

    benchmark(search)


@mark.benchmark(group="array")
def test_arrow_integer_500(rng: np.random.Generator, benchmark):
    ids = rng.choice(VOCAB_SIZE * 1000, VOCAB_SIZE, replace=False)
    index = pa.array(ids)

    query = rng.choice(ids, 500)

    def search():
        _val = pc.index_in(query, index)

    benchmark(search)


@mark.benchmark(group="single")
def test_arrow_string_single(rng: np.random.Generator, benchmark):
    ids = [str(uuid4()) for i in range(VOCAB_SIZE)]
    index = pa.array(ids)

    query = rng.choice(ids, 1).item()

    def search():
        _val = index.index(query)

    benchmark(search)


@mark.benchmark(group="array")
def test_arrow_string_500(rng: np.random.Generator, benchmark):
    ids = [str(uuid4()) for i in range(VOCAB_SIZE)]
    index = pa.array(ids)

    query = rng.choice(ids, 500)

    def search():
        _val = pc.index_in(query, index)

    benchmark(search)
