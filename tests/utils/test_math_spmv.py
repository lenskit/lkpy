# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for matrix-vector multiplication to make sure it works properly.

Bug: https://github.com/pytorch/pytorch/issues/127491
"""

import numpy as np
import scipy.sparse as sps
import torch

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import HealthCheck, given, note, settings
from pytest import approx, mark, skip


def draw_problem(
    data, nrows: int, ncols: int, dtype=np.float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # draw the initial matrix
    M = data.draw(
        nph.arrays(
            dtype,
            (nrows, ncols),
            elements=st.floats(
                -1e4,
                1e4,
                allow_nan=False,
                allow_infinity=False,
                width=np.finfo(dtype).bits,  # type: ignore
            ),
        )
    )
    # draw the vector
    v = data.draw(
        nph.arrays(
            dtype,
            ncols,
            elements=st.floats(
                -1e4,
                1e4,
                allow_nan=False,
                allow_infinity=False,
                width=np.finfo(dtype).bits,  # type: ignore
            ),
        )
    )
    # zero out items in the matrix
    mask = data.draw(nph.arrays(np.bool_, (nrows, ncols)))
    M[~mask] = 0.0
    nnz = np.sum(M != 0.0)
    note("matrix {} x {} ({} nnz)".format(nrows, ncols, nnz))

    # multiply them (dense operation with NumPy) to get expected result
    res = M @ v
    # just make sure everything's finite, should always pass due to data limits
    assert np.all(np.isfinite(res))

    return M, v, res


def tolerances(dtype=np.float64):
    # make our tolerance depend on the data type we got
    if dtype == np.float64:
        rtol, atol = 1.0e-5, 1.0e-4
    elif dtype == np.float32:
        rtol, atol = 0.05, 1.0e-3
    else:
        raise TypeError(f"unexpected data type {dtype}")
    return rtol, atol


def torchify(M, v, res, rtol, atol) -> tuple[torch.Tensor, torch.Tensor]:
    T = torch.from_numpy(M)
    tv = torch.from_numpy(v)
    assert torch.mv(T, tv).numpy() == approx(res, rel=rtol, abs=atol)
    return T, tv


def torch_test(func):
    wrapped = given(st.data(), st.integers(1, 500), st.integers(1, 500))(func)
    wrapped = settings(
        deadline=1000, max_examples=500, suppress_health_check=[HealthCheck.too_slow]
    )(wrapped)
    return wrapped


@mark.skip(reason="test unnecessary")
@torch_test
def test_torch_spmv_coo(data, nrows, ncols):
    rtol, atol = tolerances(np.float64)
    M, v, res = draw_problem(data, nrows, ncols)
    T, tv = torchify(M, v, res, rtol, atol)

    TS = T.to_sparse_coo().coalesce()

    tres = torch.mv(TS, tv)
    assert tres.numpy() == approx(res, rel=rtol, abs=atol)


@torch_test
def test_torch_spmv_csr(data, nrows, ncols):
    rtol, atol = tolerances(np.float64)
    M, v, res = draw_problem(data, nrows, ncols)
    T, tv = torchify(M, v, res, rtol, atol)

    TS = T.to_sparse_csr()

    tres = torch.mv(TS, tv)
    assert tres.numpy() == approx(res, rel=rtol, abs=atol)


@mark.skip(
    reason="spmv CSC currently broken",
)
@torch_test
def test_torch_spmv_csc(data, nrows, ncols):
    rtol, atol = tolerances(np.float64)
    M, v, res = draw_problem(data, nrows, ncols)
    T, tv = torchify(M, v, res, rtol, atol)

    TS = T.to_sparse_csc()

    tres = torch.mv(TS, tv)
    assert tres.numpy() == approx(res, rel=rtol, abs=atol)


@mark.skip("not using SPMM fallback")
@torch_test
def test_torch_spmm_coo(data, nrows, ncols):
    rtol, atol = tolerances(np.float64)
    M, v, res = draw_problem(data, nrows, ncols)
    T, tv = torchify(M, v, res, rtol, atol)

    TS = T.to_sparse_coo().coalesce()

    tres = torch.mm(TS, tv.reshape(-1, 1)).reshape(-1)
    assert tres.numpy() == approx(res, rel=rtol, abs=atol)


@mark.skip("not using SPMM fallback")
@torch_test
def test_torch_spmm_csr(data, nrows, ncols):
    rtol, atol = tolerances(np.float64)
    M, v, res = draw_problem(data, nrows, ncols)
    T, tv = torchify(M, v, res, rtol, atol)

    TS = T.to_sparse_csr()

    tres = torch.mm(TS, tv.reshape(-1, 1)).reshape(-1)
    assert tres.numpy() == approx(res, rel=rtol, abs=atol)


@mark.skip("not using SPMM fallback")
@torch_test
def test_torch_spmm_csc(data, nrows, ncols):
    rtol, atol = tolerances(np.float64)
    M, v, res = draw_problem(data, nrows, ncols)
    T, tv = torchify(M, v, res, rtol, atol)

    TS = T.to_sparse_csc()

    tres = torch.mm(TS, tv.reshape(-1, 1)).reshape(-1)
    assert tres.numpy() == approx(res, rel=rtol, abs=atol)


@mark.benchmark(max_time=10)
@mark.parametrize("device", ["cpu", "cuda"])
def test_torch_spmv_bench(device, benchmark):
    if device == "cuda" and not torch.cuda.is_available():
        skip("CUDA disabled")
    A: sps.csr_matrix = sps.random(500, 500, 0.1, format="csr")
    x = np.random.randn(500)

    T = torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr),
        torch.from_numpy(A.indices),
        torch.from_numpy(A.data),
        size=A.shape,
    ).to(device)
    v = torch.from_numpy(x).to(device)

    def op():
        torch.mv(T, v)

    benchmark(op)


@mark.benchmark(max_time=10)
@mark.parametrize("device", ["cpu", "cuda"])
def test_torch_spmm_bench(device, benchmark):
    if device == "cuda" and not torch.cuda.is_available():
        skip("CUDA disabled")
    A: sps.csr_matrix = sps.random(2500, 2500, 0.1, format="csr")
    x = np.random.randn(2500)

    T = torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr),
        torch.from_numpy(A.indices),
        torch.from_numpy(A.data),
        size=A.shape,
    ).to(device)
    v = torch.from_numpy(x).to(device)

    def op():
        torch.mm(T, v.reshape(-1, 1)).reshape(-1)

    benchmark(op)


@mark.benchmark(max_time=10)
def test_scipy_spmv_bench(benchmark):
    A: sps.csr_matrix = sps.random(2500, 2500, 0.1, format="csr")
    x = np.random.randn(2500)

    def op():
        A @ x

    benchmark(op)
