import os
import numpy as np
import scipy.linalg as sla

from pytest import approx, mark

from lenskit.math.solve import solve_tri, dposv
from lenskit.util.test import repeated


@repeated
def test_solve_ltri():
    size = np.random.randint(5, 50)
    Af = np.random.randn(size, size)
    b = np.random.randn(size)
    A = np.tril(Af)

    x = solve_tri(A, b)
    assert len(x) == size

    xexp = sla.solve_triangular(A, b, lower=True)
    assert x == approx(xexp, rel=1.0e-6)


@repeated
def test_solve_ltri_transpose():
    size = np.random.randint(5, 50)
    Af = np.random.randn(size, size)
    b = np.random.randn(size)
    A = np.tril(Af)

    x = solve_tri(A, b, True)
    assert len(x) == size

    xexp = sla.solve_triangular(A.T, b, lower=False)
    assert x == approx(xexp, rel=1.0e-6)


@repeated
def test_solve_utri():
    size = np.random.randint(5, 50)
    Af = np.random.randn(size, size)
    b = np.random.randn(size)
    A = np.triu(Af)

    x = solve_tri(A, b, lower=False)
    assert len(x) == size
    xexp = sla.solve_triangular(A, b, lower=False)
    assert x == approx(xexp, rel=1.0e-6)


@repeated
def test_solve_utri_transpose():
    size = np.random.randint(5, 50)
    Af = np.random.randn(size, size)
    b = np.random.randn(size)
    A = np.triu(Af)

    x = solve_tri(A, b, True, lower=False)
    assert len(x) == size
    xexp = sla.solve_triangular(A.T, b, lower=True)
    assert x == approx(xexp, rel=1.0e-6)


@repeated
@mark.skip(reason='we no longer use Cholesky decomposition anywhere')
def test_solve_cholesky():
    size = np.random.randint(5, 50)
    A = np.random.randn(size, size)
    b = np.random.randn(size)

    # square values of A
    A = A * A

    # and solve
    xexp, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # chol solve
    L = np.linalg.cholesky(A.T @ A)

    w = solve_tri(L, A.T @ b)
    x = solve_tri(L, w, transpose=True)

    assert x == approx(xexp, abs=1.0e-3)


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
