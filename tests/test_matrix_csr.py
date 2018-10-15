import numpy as np
import scipy.sparse as sps

import lenskit.matrix as lm

from pytest import mark, approx
import lk_test_utils as lktu


@mark.parametrize('copy', [True, False])
def test_csr_from_sps(copy):
    # initialize sparse matrix
    mat = np.random.randn(10, 5)
    mat[mat <= 0] = 0
    smat = sps.csr_matrix(mat, copy=copy)
    # make sure it's sparse
    assert smat.nnz == np.sum(mat > 0)

    csr = lm.csr_from_scipy(smat)
    assert csr.nnz == smat.nnz
    assert csr.nrows == smat.shape[0]
    assert csr.ncols == smat.shape[1]

    assert all(csr.rowptrs == smat.indptr)
    assert all(csr.colinds == smat.indices)
    assert all(csr.values == smat.data)
    assert isinstance(csr.rowptrs, np.ndarray)
    assert isinstance(csr.colinds, np.ndarray)
    assert isinstance(csr.values, np.ndarray)


def test_csr_is_numpy_compatible():
    # initialize sparse matrix
    mat = np.random.randn(10, 5)
    mat[mat <= 0] = 0
    smat = sps.csr_matrix(mat)
    # make sure it's sparse
    assert smat.nnz == np.sum(mat > 0)

    csr = lm.csr_from_scipy(smat)

    d2 = csr.values * 10
    assert d2 == approx(smat.data * 10)


def test_csr_from_coo():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = lm.csr_from_coo(rows, cols, vals)
    assert csr.nrows == 4
    assert csr.ncols == 3
    assert csr.nnz == 4
    assert csr.values == approx(vals)


def test_csr_from_coo_rand():
    for i in range(1):
        rows = np.random.randint(0, 100, 1000)
        cols = np.random.randint(0, 50, 1000)
        vals = np.random.randn(1000)

        csr = lm.csr_from_coo(rows, cols, vals, (100, 50))
        assert csr.nrows == 100
        assert csr.ncols == 50
        assert csr.nnz == 1000

        for i in range(100):
            sp = csr.rowptrs[i]
            ep = csr.rowptrs[i+1]
            assert ep - sp == np.sum(rows == i)
            points, = np.nonzero(rows == i)
            po = np.argsort(cols[points])
            points = points[po]
            assert all(np.sort(csr.colinds[sp:ep]) == cols[points])
            assert all(csr.values[np.argsort(csr.colinds[sp:ep]) + sp] == vals[points])


def test_csr_from_coo_novals():
    for i in range(50):
        rows = np.random.randint(0, 100, 1000)
        cols = np.random.randint(0, 50, 1000)

        csr = lm.csr_from_coo(rows, cols, None, (100, 50))
        assert csr.nrows == 100
        assert csr.ncols == 50
        assert csr.nnz == 1000

        for i in range(100):
            sp = csr.rowptrs[i]
            ep = csr.rowptrs[i+1]
            assert ep - sp == np.sum(rows == i)
            points, = np.nonzero(rows == i)
            po = np.argsort(cols[points])
            points = points[po]
            assert all(np.sort(csr.colinds[sp:ep]) == cols[points])
