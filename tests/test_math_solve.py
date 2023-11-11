import logging
import numpy as np
import scipy.linalg as sla

from pytest import approx

from lenskit.math.solve import dposv

from hypothesis import given, settings
import hypothesis.strategies as st

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


@settings(deadline=None)
@given(square_problem())
def test_solve_dposv(problem):
    A, b, size = problem

    # square values of A
    A = A * A

    # and solve
    xexp, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)

    F = A.T @ A
    x = A.T @ b
    dposv(F, x, True)

    assert x == approx(xexp, rel=1.0e-3)
