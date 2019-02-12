import numpy as np
import scipy.sparse as sps

from pytest import mark, approx

import lenskit.matrix as lm
mkl_ops = lm.mkl_ops()


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
