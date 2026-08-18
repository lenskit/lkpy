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
