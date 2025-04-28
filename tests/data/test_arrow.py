# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pyarrow as pa

from lenskit.data.arrow import is_sorted


def test_empty_sorted():
    tbl = pa.Table.from_pandas(pd.DataFrame({"x": []}))
    assert is_sorted(tbl, ["x"])


def test_range_sorted():
    tbl = pa.Table.from_pandas(pd.DataFrame({"x": np.arange(10)}))
    assert is_sorted(tbl, ["x"])


def test_small_unsorted():
    tbl = pa.Table.from_pandas(pd.DataFrame({"x": [5, 3]}))
    assert not is_sorted(tbl, ["x"])


def test_random_unsorted():
    tbl = pa.Table.from_pandas(pd.DataFrame({"x": np.random.randn(1000)}))
    assert not is_sorted(tbl, ["x"])


def test_two_unsorted():
    tbl = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "x": np.repeat(np.arange(100), 10),
                "y": np.random.randn(1000),
            }
        )
    )
    assert is_sorted(tbl, ["x"])
    assert not is_sorted(tbl, ["x", "y"])


def test_two_sorted():
    tbl = pa.Table.from_pandas(
        pd.concat([pd.DataFrame({"x": i, "y": np.arange(5)}) for i in range(20)], ignore_index=True)
    )
    assert is_sorted(tbl, ["x", "y"])
