# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import hypothesis.strategies as st
from hypothesis import given

from lenskit._accel import data


@given(st.lists(st.integers(0, 1000)), st.integers(0, 5000))
def test_neg_mask(indices, n):
    iarr = pa.array(indices, pa.int32())
    imax = pc.max(iarr).as_py()
    if imax is not None and imax >= n:
        n = imax + 1

    mask = data.negative_mask(n, iarr)
    np_mask = mask.to_numpy(zero_copy_only=False)

    assert len(mask) == n
    assert np.sum(np_mask) == n - len(set(indices))
    assert np.all(~np_mask[indices])
