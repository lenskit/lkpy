import numpy as np
import scipy.sparse as sps

from pytest import mark, approx

import lenskit.matrix as lm
mkl_ops = lm.mkl_ops()


@mark.skipif(mkl_ops is None, reason='MKL not available')
def test_mkl_syrk():
    for i in range(50):
        M = np.random.randn(10, 5)
        M[M <= 0] = 0
        s = sps.csr_matrix(M)
        assert s.nnz == np.sum(M > 0)

        csr = lm.csr_from_scipy(s)

        ctc = mkl_ops.csr_syrk(csr)
        res = lm.csr_to_scipy(ctc).toarray()
        res = res.T + res
        rd = np.diagonal(res)
        res = res - np.diagflat(rd) * 0.5

        mtm = M.T @ M
        assert res == approx(mtm)
