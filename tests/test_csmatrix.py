import numpy as np
import scipy.sparse as sps
from numba import njit

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import assume, given, settings
from pytest import mark

from lenskit.util.csmatrix import CSMatrix


@st.composite
def sparse_matrices(draw, max_shape=(50, 50), density=st.floats(0, 1), format="csr"):
    ubr, ubc = max_shape

    rows = draw(st.integers(1, ubr))
    cols = draw(st.integers(1, ubc))
    dens = draw(density)
    prod = rows * cols
    nnz = int(prod * dens)

    points = draw(nph.arrays("int32", nnz, elements=st.integers(0, prod - 1), unique=True))
    values = draw(nph.arrays("float64", nnz))
    rvs = points % rows
    cvs = points // rows
    assert np.all(rvs < rows)
    assert np.all(cvs < cols)

    return sps.csr_array((values, (rvs, cvs)), shape=(rows, cols))


@given(sparse_matrices())
def test_init_matrix(m: sps.csr_array):
    print(m.shape, m.nnz, m.indptr.dtype, m.indices.dtype)
    nr, nc = m.shape

    m2 = CSMatrix(nr, nc, m.indptr, m.indices, m.data)

    assert m2.nrows == nr
    assert m2.ncols == nc
    assert m2.nnz == m.nnz


@given(sparse_matrices())
def test_from_scipy(m: sps.csr_array):
    print(m.shape, m.nnz, m.indptr.dtype, m.indices.dtype)
    m2 = CSMatrix.from_scipy(m)

    assert m2.nrows == m.shape[0]
    assert m2.ncols == m.shape[1]
    assert m2.nnz == m.nnz


@given(sparse_matrices())
def test_csm_row_ep(m: sps.csr_array):
    m2 = CSMatrix.from_scipy(m)

    for i in range(m2.nrows):
        sp, ep = m2.row_ep(i)
        assert sp == m2.rowptr[i]
        assert ep == m2.rowptr[i + 1]
