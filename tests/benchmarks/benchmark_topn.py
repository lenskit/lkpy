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

from lenskit._accel import data
from lenskit.data.types import argtopn
from lenskit.parallel import ensure_parallel_init


@mark.benchmark(group="all")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_numpy_argsort(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)
    scores = np.require(scores, dtype=np.float32)

    def sort():
        _idx = np.argsort(-scores)

    benchmark(sort)


@mark.benchmark(group="all")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_torch_argsort(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)
    scores = torch.as_tensor(np.require(scores, dtype=np.float32))

    def sort():
        _idx = torch.argsort(-scores)

    benchmark(sort)


@mark.benchmark(group="all")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_argsort_gpu(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)
    scores = torch.as_tensor(np.require(scores, dtype=np.float32)).cuda()

    def sort():
        _idx = torch.argsort(-scores).cpu()

    benchmark(sort)


@mark.benchmark(group="all")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
@mark.skip("only testing argsort")
def test_numpy_sort(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)
    scores = np.require(scores, dtype=np.float32)

    def sort():
        _res = np.sort(-scores)

    benchmark(sort)


@mark.benchmark(group="all")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_arrow_sort(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)
    scores = pa.array(scores, pa.float32())

    def sort():
        _idx = pc.array_sort_indices(scores, order="descending")

    benchmark(sort)


@mark.benchmark(group="all")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_accel_sort(rng: np.random.Generator, size: int, benchmark):
    ensure_parallel_init()
    scores = rng.standard_exponential(size)
    scores = pa.array(scores, pa.float32())

    def sort():
        _idx = data.argsort_descending(scores)

    benchmark(sort)


@mark.benchmark(group="topn")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_argtopn(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)

    def sort():
        _res = argtopn(scores, 100)

    benchmark(sort)


@mark.benchmark(group="topn")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
def test_torch_topk(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)
    scores = torch.as_tensor(scores)

    def sort():
        _idx = torch.topk(-scores, 100)

    benchmark(sort)


@mark.benchmark(group="topn")
@mark.parametrize("size", [100, 5000, 100_000, 1_000_000])
@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_topk_gpu(rng: np.random.Generator, size: int, benchmark):
    scores = rng.standard_exponential(size)
    scores = torch.as_tensor(scores).cuda()

    def sort():
        _s, _idx = torch.topk(-scores, 100)
        _s = _s.cpu()
        _idx = _idx.cpu()

    benchmark(sort)
