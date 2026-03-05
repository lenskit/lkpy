# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def test_null_mixup(rng: np.random.Generator):
    """
    Test for Arrow bug 47234.
    """
    nums = rng.integers(0, 10_000, size=100)
    mask = np.zeros(100, dtype=np.bool_)
    mask[-1] = True

    arr = pa.array(nums, mask=mask)
    print(arr)
    assert arr.null_count == 1

    a2 = pc.fill_null(arr, -1)
    print(a2)
    assert a2.null_count == 0

    npa2 = a2.to_numpy()
    assert npa2[-1] < 0
    assert np.all(npa2[:-1] >= 0)
