# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scipy.sparse import csr_array

from pytest import importorskip, mark

from lenskit.data import Dataset
from lenskit.testing import ml_20m

numba = importorskip("numba")


@mark.benchmark()
def test_select_arrow(ml_20m: Dataset, rng: np.random.Generator, benchmark):
    matrix = ml_20m.interactions().matrix().scipy(layout="csr")

    offsets = np.require(matrix.indptr[:-1], np.int32)
    offsets = pa.array(offsets)
    pairs = pa.StructArray.from_arrays(
        [pa.array(matrix.indices), pa.array(matrix.data)], ["column", "value"]
    )
    lists = pa.ListArray.from_arrays(offsets, pairs)

    users = rng.choice(ml_20m.user_count, 1000, replace=True)
    users = np.require(users, "i4")
    users = pa.array(users)

    def select():
        _res = pc.array_take(lists, users, boundscheck=False)

    benchmark(select)


@mark.benchmark()
def test_select_scipy_csr(ml_20m: Dataset, rng: np.random.Generator, benchmark):
    matrix = ml_20m.interactions().matrix().scipy(layout="csr")

    users = rng.choice(ml_20m.user_count, 1000, replace=True)
    users = np.require(users, "i4")

    def select():
        _res = matrix[users, :]

    benchmark(select)


@mark.benchmark()
def test_select_numba(ml_20m: Dataset, rng: np.random.Generator, benchmark):
    matrix = ml_20m.interactions().matrix().scipy(layout="csr")

    users = rng.choice(ml_20m.user_count, 1000, replace=True)
    users = np.require(users, "i4")
    numba_take(matrix, users)

    def select():
        _res = numba_take(matrix, users)

    benchmark(select)


def numba_take(matrix: csr_array, rows: np.ndarray[tuple[int], np.dtype[np.int32]]):
    assert matrix.shape is not None

    _nr, nc = matrix.shape
    rp2, ci2, vs2 = _select_matrix(matrix.indptr, matrix.indices, matrix.data, rows)
    return csr_array((vs2, ci2, rp2), shape=(len(rows), nc))


@numba.njit
def _select_matrix(
    rps: np.ndarray[tuple[int], np.dtype[np.int32]],
    cis: np.ndarray[tuple[int], np.dtype[np.int32]],
    vs: np.ndarray[tuple[int], np.dtype[np.float32]],
    rows: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.float32]],
]:
    ptrs = np.zeros(len(rows) + 1, np.int32)
    tot = 0
    for i, r in enumerate(rows):
        sp = rps[r]
        ep = rps[r + 1]
        rl = ep - sp
        ptrs[i + 1] = ptrs[i] + rl
        tot += rl

    inds = np.empty(tot, np.int32)
    vals = np.empty(tot, np.float32)

    for i, r in enumerate(rows):
        sp = rps[r]
        ep = rps[r + 1]

        spo = ptrs[i]
        epo = ptrs[i + 1]
        inds[spo:epo] = cis[sp:ep]
        vals[spo:epo] = vs[sp:ep]

    return ptrs, inds, vals
