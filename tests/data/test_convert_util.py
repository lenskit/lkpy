# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

from lenskit.data._adapt import py_scalar


def test_py_scalar_pyint():
    assert py_scalar(42) == 42


def test_py_scalar_pystr():
    assert py_scalar("foo") == "foo"


def test_py_scalar_npint():
    x = py_scalar(np.int32(100))
    assert x == 100
    assert isinstance(x, int)
