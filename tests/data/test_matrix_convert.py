# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pyarrow as pa
from scipy.sparse import csr_array

from hypothesis import given

from lenskit.data.matrix import sparse_from_arrow, sparse_to_arrow
from lenskit.testing import sparse_arrays


@given(sparse_arrays())
def test_to_arrow(csr: csr_array):
    arr = sparse_to_arrow(csr)

    assert pa.types.is_list(arr.type)
    assert len(arr) == csr.shape[0]
    assert len(arr.values) == csr.nnz
    assert np.all(arr.offsets.to_numpy() == csr.indptr)


@given(sparse_arrays())
def test_from_arrow(csr: csr_array):
    arr = sparse_to_arrow(csr)

    csr2 = sparse_from_arrow(arr, csr.shape)  # type: ignore
    assert csr2.shape == csr.shape
    assert csr2.nnz == csr.nnz
    assert np.all(csr2.indptr == csr.indptr)
    assert np.all(csr2.indices == csr.indices)
    assert np.all(csr2.data == csr.data.astype("f4"))
