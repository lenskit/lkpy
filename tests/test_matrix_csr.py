from itertools import product
import numpy as np
import scipy.sparse as sps

from lenskit import sharing
import lenskit.matrix as lm
import lk_test_utils as lktu

from pytest import mark, approx


@mark.parametrize('copy', [True, False])
def test_csr_from_sps(copy):
    # initialize sparse matrix
    mat = np.random.randn(10, 5)
    mat[mat <= 0] = 0
    smat = sps.csr_matrix(mat)
    # make sure it's sparse
    assert smat.nnz == np.sum(mat > 0)

    csr = lm.csr_from_scipy(smat, copy=copy)
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


def test_csr_row():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_) + 1

    csr = lm.csr_from_coo(rows, cols, vals)
    assert all(csr.row(0) == np.array([0, 1, 2], dtype=np.float_))
    assert all(csr.row(1) == np.array([3, 0, 0], dtype=np.float_))
    assert all(csr.row(2) == np.array([0, 0, 0], dtype=np.float_))
    assert all(csr.row(3) == np.array([0, 4, 0], dtype=np.float_))


def test_csr_sparse_row():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = lm.csr_from_coo(rows, cols, vals)
    assert all(csr.row_cs(0) == np.array([1, 2], dtype=np.int32))
    assert all(csr.row_cs(1) == np.array([0], dtype=np.int32))
    assert all(csr.row_cs(2) == np.array([], dtype=np.int32))
    assert all(csr.row_cs(3) == np.array([1], dtype=np.int32))

    assert all(csr.row_vs(0) == np.array([0, 1], dtype=np.float_))
    assert all(csr.row_vs(1) == np.array([2], dtype=np.float_))
    assert all(csr.row_vs(2) == np.array([], dtype=np.float_))
    assert all(csr.row_vs(3) == np.array([3], dtype=np.float_))


def test_csr_transpose():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = lm.csr_from_coo(rows, cols, vals)
    csc = csr.transpose()
    assert csc.nrows == csr.ncols
    assert csc.ncols == csr.nrows

    assert all(csc.rowptrs == [0, 1, 3, 4])
    assert csc.colinds.max() == 3
    assert csc.values.sum() == approx(vals.sum())

    for r, c, v in zip(rows, cols, vals):
        row = csc.row(c)
        assert row[r] == v


def test_csr_transpose_coords():
    rows = np.array([0, 0, 1, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 1], dtype=np.int32)
    vals = np.arange(4, dtype=np.float_)

    csr = lm.csr_from_coo(rows, cols, vals)
    csc = csr.transpose_coords()
    assert csc.nrows == csr.ncols
    assert csc.ncols == csr.nrows

    assert all(csc.rowptrs == [0, 1, 3, 4])
    assert csc.colinds.max() == 3
    assert csc.values is None

    for r, c, v in zip(rows, cols, vals):
        row = csc.row(c)
        assert row[r] == 1


def test_csr_row_nnzs():
    # initialize sparse matrix
    mat = np.random.randn(10, 5)
    mat[mat <= 0] = 0
    smat = sps.csr_matrix(mat)
    # make sure it's sparse
    assert smat.nnz == np.sum(mat > 0)
    csr = lm.csr_from_scipy(smat)

    nnzs = csr.row_nnzs()
    assert nnzs.sum() == csr.nnz
    for i in range(10):
        row = mat[i, :]
        assert nnzs[i] == np.sum(row > 0)


def test_csr_from_coo_rand():
    for i in range(100):
        coords = np.random.choice(np.arange(50 * 100, dtype=np.int32), 1000, False)
        rows = np.mod(coords, 100, dtype=np.int32)
        cols = np.floor_divide(coords, 100, dtype=np.int32)
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
            assert len(points) == ep - sp
            po = np.argsort(cols[points])
            points = points[po]
            assert all(np.sort(csr.colinds[sp:ep]) == cols[points])
            assert all(np.sort(csr.row_cs(i)) == cols[points])
            assert all(csr.values[np.argsort(csr.colinds[sp:ep]) + sp] == vals[points])
            row = np.zeros(50)
            row[cols[points]] = vals[points]
            assert np.sum(csr.row(i)) == approx(np.sum(vals[points]))
            assert all(csr.row(i) == row)


def test_csr_from_coo_novals():
    for i in range(50):
        coords = np.random.choice(np.arange(50 * 100, dtype=np.int32), 1000, False)
        rows = np.mod(coords, 100, dtype=np.int32)
        cols = np.floor_divide(coords, 100, dtype=np.int32)

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
            assert np.sum(csr.row(i)) == len(points)


def test_csr_to_sps():
    # initialize sparse matrix
    mat = np.random.randn(10, 5)
    mat[mat <= 0] = 0
    # get COO
    smat = sps.coo_matrix(mat)
    # make sure it's sparse
    assert smat.nnz == np.sum(mat > 0)

    csr = lm.csr_from_coo(smat.row, smat.col, smat.data, shape=smat.shape)
    assert csr.nnz == smat.nnz
    assert csr.nrows == smat.shape[0]
    assert csr.ncols == smat.shape[1]

    smat2 = lm.csr_to_scipy(csr)
    assert sps.isspmatrix(smat2)
    assert sps.isspmatrix_csr(smat2)

    for i in range(csr.nrows):
        assert smat2.indptr[i] == csr.rowptrs[i]
        assert smat2.indptr[i+1] == csr.rowptrs[i+1]
        sp = smat2.indptr[i]
        ep = smat2.indptr[i+1]
        assert all(smat2.indices[sp:ep] == csr.colinds[sp:ep])
        assert all(smat2.data[sp:ep] == csr.values[sp:ep])


@mark.parametrize("prefix,values", product([None, 'p_'], [True, False]))
def test_csr_save_load(tmp_path, prefix, values):
    tmp_path = lktu.norm_path(tmp_path)
    coords = np.random.choice(np.arange(50 * 100, dtype=np.int32), 1000, False)
    rows = np.mod(coords, 100, dtype=np.int32)
    cols = np.floor_divide(coords, 100, dtype=np.int32)
    if values:
        vals = np.random.randn(1000)
    else:
        vals = None

    csr = lm.csr_from_coo(rows, cols, vals, (100, 50))
    assert csr.nrows == 100
    assert csr.ncols == 50
    assert csr.nnz == 1000

    data = lm.csr_save(csr, prefix=prefix)

    np.savez_compressed(tmp_path / 'matrix.npz', **data)

    with np.load(tmp_path / 'matrix.npz') as npz:
        csr2 = lm.csr_load(npz, prefix=prefix)

    assert csr2.nrows == csr.nrows
    assert csr2.ncols == csr.ncols
    assert csr2.nnz == csr.nnz
    assert all(csr2.rowptrs == csr.rowptrs)
    assert all(csr2.colinds == csr.colinds)
    if values:
        assert all(csr2.values == csr.values)
    else:
        assert csr2.values is None

@mark.parametrize("share,prefix,values", product(sharing.share_impls, [None, 'p_'], [True, False]))
def test_csr_share(share, prefix, values):
    coords = np.random.choice(np.arange(50 * 100, dtype=np.int32), 1000, False)
    rows = np.mod(coords, 100, dtype=np.int32)
    cols = np.floor_divide(coords, 100, dtype=np.int32)
    if values:
        vals = np.random.randn(1000)
    else:
        vals = None

    csr = lm.csr_from_coo(rows, cols, vals, (100, 50))
    assert csr.nrows == 100
    assert csr.ncols == 50
    assert csr.nnz == 1000

    with share() as ctx:
        key = lm.csr_share_publish(csr, ctx)
        csr2 = lm.csr_share_resolve(key, ctx)

        assert csr2.nrows == csr.nrows
        assert csr2.ncols == csr.ncols
        assert csr2.nnz == csr.nnz
        assert all(csr2.rowptrs == csr.rowptrs)
        assert all(csr2.colinds == csr.colinds)
        if values:
            assert all(csr2.values == csr.values)
        else:
            assert csr2.values is None
