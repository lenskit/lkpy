# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import HealthCheck, assume, given, settings
from pytest import approx, mark

from lenskit.data import sparse_ratings
from lenskit.data.matrix import safe_spmv, torch_sparse_from_scipy
from lenskit.util.test import coo_arrays, ml_test

_log = logging.getLogger(__name__)


def test_sparse_ratings(rng):
    ratings = ml_test.ratings
    mat, uidx, iidx = sparse_ratings(ratings)

    assert mat.nrows == len(uidx)
    assert mat.nrows == ratings.user.nunique()
    assert mat.ncols == len(iidx)
    assert mat.ncols == ratings.item.nunique()

    # user indicators should correspond to user item counts
    ucounts = ratings.groupby("user").item.count()
    ucounts = ucounts.loc[uidx].cumsum()
    assert all(mat.rowptrs[1:] == ucounts.values)

    # verify rating values
    ratings = ratings.set_index(["user", "item"])
    for u in rng.choice(uidx, size=50):
        ui = uidx.get_loc(u)
        vs = mat.row_vs(ui)
        vs = pd.Series(vs, iidx[mat.row_cs(ui)])
        rates = ratings.loc[u]["rating"]
        vs, rates = vs.align(rates)
        assert not any(vs.isna())
        assert not any(rates.isna())
        assert all(vs == rates)


def test_sparse_ratings_implicit():
    ratings = ml_test.ratings
    ratings = ratings.loc[:, ["user", "item"]]
    mat, uidx, iidx = sparse_ratings(ratings)

    assert mat.nrows == len(uidx)
    assert mat.nrows == ratings.user.nunique()
    assert mat.ncols == len(iidx)
    assert mat.ncols == ratings.item.nunique()
    assert mat.values is None


@mark.parametrize(
    "format, sps_fmt_checker",
    [
        (True, sps.isspmatrix_csr),
        ("csr", sps.isspmatrix_csr),
        ("coo", sps.isspmatrix_coo),
    ],
)
def test_sparse_ratings_scipy(format, sps_fmt_checker):
    ratings = ml_test.ratings
    mat, uidx, iidx = sparse_ratings(ratings, scipy=format)

    assert sps.issparse(mat)
    assert sps_fmt_checker(mat)
    assert len(uidx) == ratings.user.nunique()
    assert len(iidx) == ratings.item.nunique()

    # user indicators should correspond to user item counts
    ucounts = ratings.groupby("user").item.count()
    ucounts = ucounts.loc[uidx].cumsum()
    if sps.isspmatrix_coo(mat):
        mat = mat.tocsr()
    assert all(mat.indptr[1:] == ucounts.values)


def test_sparse_ratings_scipy_implicit():
    ratings = ml_test.ratings
    ratings = ratings.loc[:, ["user", "item"]]
    mat, uidx, iidx = sparse_ratings(ratings, scipy=True)

    assert sps.issparse(mat)
    assert sps.isspmatrix_csr(mat)
    assert len(uidx) == ratings.user.nunique()
    assert len(iidx) == ratings.item.nunique()

    assert all(mat.data == 1.0)


def test_sparse_ratings_torch():
    ratings = ml_test.ratings
    mat: torch.Tensor
    mat, uidx, iidx = sparse_ratings(ratings, torch=True)

    assert torch.is_tensor(mat)
    assert mat.is_sparse_csr
    assert len(uidx) == ratings.user.nunique()
    assert len(iidx) == ratings.item.nunique()


def test_sparse_ratings_indexes(rng):
    ratings = ml_test.ratings
    uidx = pd.Index(rng.permutation(ratings["user"].unique()))
    iidx = pd.Index(rng.permutation(ratings["item"].unique()))

    mat, _uidx, _iidx = sparse_ratings(ratings, users=uidx, items=iidx)

    assert _uidx is uidx
    assert _iidx is iidx
    assert len(_uidx) == ratings.user.nunique()
    assert len(_iidx) == ratings.item.nunique()

    # verify rating values
    ratings = ratings.set_index(["user", "item"])
    for u in rng.choice(_uidx, size=50):
        ui = _uidx.get_loc(u)
        vs = mat.row_vs(ui)
        vs = pd.Series(vs, _iidx[mat.row_cs(ui)])
        rates = ratings.loc[u]["rating"]
        vs, rates = vs.align(rates)
        assert not any(vs.isna())
        assert not any(rates.isna())
        assert all(vs == rates)


@settings(deadline=1000, max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(st.data())
def test_safe_spmv(data):
    M = data.draw(coo_arrays(dtype="f8", shape=st.tuples(st.integers(1, 500), st.integers(1, 500))))
    nr, nc = M.shape
    v = data.draw(
        nph.arrays(
            M.data.dtype,
            nc,
            elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False, width=32),
        )
    )
    res = M @ v
    assume(np.all(np.isfinite(res)))

    TM = torch_sparse_from_scipy(M, "csr")
    tv = torch.from_numpy(v)

    tres = safe_spmv(TM, tv)
    assert tres.cpu().numpy() == approx(res, rel=1.0e-4, abs=1.0e-5)
