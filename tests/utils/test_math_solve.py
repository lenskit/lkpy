# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import torch

import hypothesis.strategies as st
from hypothesis import given
from pytest import approx

from lenskit.math.solve import solve_cholesky

_log = logging.getLogger(__name__)


@st.composite
def square_problem(draw, scale=10):
    size = draw(st.integers(2, 100))

    # Hypothesis doesn't do well at generating problem data, so go with this
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.RandomState(seed)
    A = rng.randn(size, size) * scale
    b = rng.randn(size) * scale
    return A, b, size


@given(square_problem())
def test_solve_cholesky(problem):
    A, b, size = problem

    # square values of A so it's positive
    A = A * A

    # and solve least squares
    xexp, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)

    F = A.T @ A
    y = A.T @ b
    x = solve_cholesky(F, y)
    assert x.shape == y.shape

    assert x == approx(xexp, rel=1.0e-3)

    assert F @ x == approx(y, rel=2.0e-6, abs=5.0e-9)
