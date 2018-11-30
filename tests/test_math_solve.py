import numpy as np
import scipy.linalg as sla

from pytest import approx

from lenskit.math.solve import *


def test_solve_ltri():
    for i in range(10):
        size = np.random.randint(5, 50)
        Af = np.random.randn(size, size)
        b = np.random.randn(size)
        A = np.tril(Af)

        x = solve_ltri(A, b)
        assert len(x) == size

        xexp = sla.solve_triangular(A, b, lower=True)
        assert x == approx(xexp, rel=1.0e-6)


def test_solve_ltri_transpose():
    for i in range(10):
        size = np.random.randint(5, 50)
        Af = np.random.randn(size, size)
        b = np.random.randn(size)
        A = np.tril(Af)

        x = solve_ltri(A, b, True)
        assert len(x) == size

        xexp = sla.solve_triangular(A.T, b, lower=False)
        assert x == approx(xexp, rel=1.0e-6)


def test_solve_utri():
    for i in range(10):
        size = np.random.randint(5, 50)
        Af = np.random.randn(size, size)
        b = np.random.randn(size)
        A = np.triu(Af)

        x = solve_utri(A, b)
        assert len(x) == size
        xexp = sla.solve_triangular(A, b, lower=False)
        assert x == approx(xexp, rel=1.0e-6)


def test_solve_utri_transpose():
    for i in range(10):
        size = np.random.randint(5, 50)
        Af = np.random.randn(size, size)
        b = np.random.randn(size)
        A = np.triu(Af)

        x = solve_utri(A, b, True)
        assert len(x) == size
        xexp = sla.solve_triangular(A.T, b, lower=True)
        assert x == approx(xexp, rel=1.0e-6)


def test_solve_cholesky():
    for i in range(10):
        size = np.random.randint(5, 50)
        A = np.random.randn(size, size)
        b = np.random.randn(size)

        # square values of A
        A = A * A

        # and solve
        xexp, resid, rank, s = np.linalg.lstsq(A, b)

        # chol solve
        L = np.linalg.cholesky(A.T @ A)

        w = solve_ltri(L, A.T @ b)
        x = solve_ltri(L, w, True)

        assert x == approx(xexp)
