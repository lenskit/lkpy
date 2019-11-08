import os
import numpy as np
import scipy.sparse as sps
import cffi

from pytest import mark, approx, skip

import lenskit.matrix as lm
import lenskit.util.test as lktu

mkl_ops = lm.mkl_ops()


def test_mkl_available():
    if 'CONDA_PREFIX' in os.environ:
        assert mkl_ops is not None
        assert mkl_ops.clib is not None
    else:
        skip('only require MKL availability in Conda')


@mark.skipif(mkl_ops is None, reason='MKL not available')
def test_mkl_mult_vec():
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


@mark.skipif(mkl_ops is None, reason='MKL not available')
def test_mkl_syrk():
    for i in range(50):
        M = np.random.randn(10, 5)
        M[M <= 0] = 0
        s = sps.csr_matrix(M)
        assert s.nnz == np.sum(M > 0)

        csr = lm.CSR.from_scipy(s)

        ctc = mkl_ops.csr_syrk(csr)
        res = ctc.to_scipy().toarray()
        res = res.T + res
        rd = np.diagonal(res)
        res = res - np.diagflat(rd) * 0.5

        mtm = M.T @ M
        assert res == approx(mtm)


@mark.skipif(mkl_ops is None, reason='MKL not available')
def test_mkl_mabt():
    for i in range(50):
        A = lktu.rand_csr(20, 10, nnz=50)
        B = lktu.rand_csr(5, 10, nnz=20)

        As = mkl_ops.SparseM.from_csr(A)
        Bs = mkl_ops.SparseM.from_csr(B)

        Ch = mkl_ops._lk_mkl_spmabt(As.ptr, Bs.ptr)
        C = mkl_ops._to_csr(Ch)
        C = lm.CSR(N=C)

        assert C.nrows == 20
        assert C.ncols == 5

        Csp = A.to_scipy() @ B.to_scipy().T
        Cspa = Csp.toarray()
        Ca = C.to_scipy().toarray()
        assert Ca == approx(Cspa)
