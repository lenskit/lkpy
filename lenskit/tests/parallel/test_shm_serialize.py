# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import torch

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import assume, given, settings

from lenskit.parallel.serialize import shm_deserialize, shm_serialize
from lenskit.testing import sparse_tensors


@given(st.lists(st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False))))
def test_basic_shm(obj):
    data = shm_serialize(obj)
    o2 = shm_deserialize(data)

    assert o2 == obj


@given(nph.arrays(nph.scalar_dtypes(), st.integers(0, 5000)))
def test_share_ndarray(arr):
    assume(np.all(np.isfinite(arr)))
    data = shm_serialize(arr)
    a2 = shm_deserialize(data)

    assert a2.shape == arr.shape
    assert np.all(a2 == arr)


@settings(deadline=5000)
@given(
    nph.arrays(
        st.one_of(nph.integer_dtypes(endianness="="), nph.floating_dtypes(endianness="=")),
        st.integers(0, 5000),
    )
)
def test_share_torch(arr):
    assume(np.all(np.isfinite(arr)))
    arr = torch.from_numpy(arr)
    data = shm_serialize(arr)
    a2 = shm_deserialize(data)

    assert a2.shape == arr.shape
    assert torch.all(a2 == arr)


@settings(deadline=1000)
@given(sparse_tensors(layout=["csr", "coo", "csc"]))
def test_share_sparse_torch(arr):
    data = shm_serialize(arr)
    a2 = shm_deserialize(data)

    assert a2.shape == arr.shape
    assert a2.layout == arr.layout
    assert torch.all(a2.to_dense() == arr.to_dense())
