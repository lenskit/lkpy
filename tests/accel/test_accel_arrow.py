# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Test Arrow utility functions in the accelerator.
"""

import numpy as np
import pyarrow as pa

from lenskit import _accel


def test_array_type():
    arr = pa.array(np.arange(10, dtype=np.int32))
    t = _accel.arrow_type(arr)
    assert t == "Int32"
