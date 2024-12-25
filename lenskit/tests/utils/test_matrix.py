# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import torch

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import HealthCheck, assume, given, settings
from pytest import approx

from lenskit.math.sparse import safe_spmv, torch_sparse_from_scipy
from lenskit.testing import coo_arrays


@settings(deadline=1000, suppress_health_check=[HealthCheck.too_slow])
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
    assert tres.cpu().numpy() == approx(res, rel=1.0e-3, abs=1.0e-5)
