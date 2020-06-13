import os
import numpy as np
import scipy.linalg as sla

from pytest import approx, mark

from lenskit.math.solve import dposv
from lenskit.util.test import repeated


@repeated
def test_solve_dposv():
    size = np.random.randint(5, 50)
    A = np.random.randn(size, size)
    b = np.random.randn(size)

    # square values of A
    A = A * A

    # and solve
    xexp, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)

    F = A.T @ A
    x = A.T @ b
    dposv(F, x, True)

    assert x == approx(xexp, rel=1.0e-3)
