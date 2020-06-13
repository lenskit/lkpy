import os
import numpy as np
import scipy.sparse as sps

from pytest import approx, skip, fixture
from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

import lenskit.matrix as lm
import lenskit.util.test as lktu


@fixture(scope='module')
def mkl_ops():
    ops = lm.mkl_ops()
    if ops is None:
        skip('MKL not available')
    return ops


def test_mkl_available():
    if 'CONDA_PREFIX' in os.environ:
        ops = lm.mkl_ops()
        assert ops is not None
        assert ops.clib is not None
    else:
        skip('only require MKL availability in Conda')


def test_mkl_mult_vec(mkl_ops):
    for i in range(50):
        m = np.random.randint(5, 100)
        n = np.random.randint(5, 100)

        M = np.random.randn(m, n)
        M[M <= 0] = 0
        s = sps.csr_matrix(M)
        assert s.nnz == np.sum(M > 0)

        csr = lm.CSR.from_scipy(s)
        mklM = mkl_ops.SparseM.from_csr(csr)

        x = np.random.randn(n)

        y = np.zeros(m)
        y = mklM.mult_vec(1, x, 0, y)
        assert len(y) == m

        y2 = s @ x

        assert y == approx(y2)


@settings(deadline=None)
@given(st.data())
def test_mkl_mabt(mkl_ops, data):
    Anr = data.draw(st.integers(5, 100))
    nc = data.draw(st.integers(5, 100))
    Bnr = data.draw(st.integers(5, 100))
    A = data.draw(lktu.csrs(nrows=Anr, ncols=nc, values=True))
    B = data.draw(lktu.csrs(nrows=Bnr, ncols=nc, values=True))

    As = mkl_ops.SparseM.from_csr(A)
    Bs = mkl_ops.SparseM.from_csr(B)

    Ch = mkl_ops._lk_mkl_spmabt(As.ptr, Bs.ptr)
    C = mkl_ops._to_csr(Ch)
    C = lm.CSR(N=C)

    assert C.nrows == A.nrows
    assert C.ncols == B.nrows

    Csp = A.to_scipy() @ B.to_scipy().T
    Cspa = Csp.toarray()
    Ca = C.to_scipy().toarray()
    assert Ca == approx(Cspa)
