"""
Utilities for working with rating matrices.
"""

from collections import namedtuple
import logging
import warnings

import pandas as pd
import numpy as np
import scipy.sparse as sps
import numba as n
from numba import njit, jitclass, objmode

_logger = logging.getLogger(__name__)

RatingMatrix = namedtuple('RatingMatrix', ['matrix', 'users', 'items'])


@jitclass({
    'nrows': n.int32,
    'ncols': n.int32,
    'nnz': n.int32,
    'rowptrs': n.int32[:],
    'colinds': n.int32[:],
    'values': n.optional(n.float64[:])
})
class CSR:
    """
    Simple compressed sparse row matrix.  This is like :py:class:`scipy.sparse.csr_matrix`, with
    a couple of useful differences:

    * It is a Numba jitclass, so it can be directly used from Numba-optimized functions.
    * The value array is optional, for cases in which only the matrix structure is required.
    * The value array, if present, is always double-precision.

    You generally don't want to create this class yourself.  Instead, use one of the related
    utility functions.

    Attributes:
        nrows(int): the number of rows.
        ncols(int): the number of columns.
        nnz(int): the number of entries.
        rowptrs(numpy.ndarray): the row pointers.
        colinds(numpy.ndarray): the column indices.
        values(numpy.ndarray): the values
    """
    def __init__(self, nrows, ncols, nnz, ptrs, inds, vals):
        self.nrows = nrows
        self.ncols = ncols
        self.nnz = nnz
        self.rowptrs = ptrs
        self.colinds = inds
        self.values = vals

    def row(self, row):
        sp = self.rowptrs[row]
        ep = self.rowptrs[row + 1]

        v = np.zeros(self.ncols)
        cols = self.colinds[sp:ep]
        if self.values is None:
            v[cols] = 1
        else:
            v[cols] = self.values[sp:ep]

        return v

    def row_cs(self, row):
        sp = self.rowptrs[row]
        ep = self.rowptrs[row + 1]

        return self.colinds[sp:ep]

    def row_vs(self, row):
        sp = self.rowptrs[row]
        ep = self.rowptrs[row + 1]

        if self.values is None:
            return np.full(ep - sp, 1.0)
        else:
            return self.values[sp:ep]

    def row_nnzs(self):
        "Get a vector of the number of nonzero entries in each row."
        # we want to use np.diff, but numba doesn't like it with our rowptr
        diff = np.zeros(self.nrows, dtype=np.int32)
        for i in range(self.nrows):
            diff[i] = self.rowptrs[i+1] - self.rowptrs[i]
        return diff

    def sort_values(self):
        for i in range(self.nrows):
            sp = self.rowptrs[i]
            ep = self.rowptrs[i+1]
            if ep == sp:
                continue

            ord = np.argsort(self.values[sp:ep])
            ord = ord[::-1]
            self.colinds[sp:ep] = self.colinds[sp + ord]
            self.values[sp:ep] = self.values[sp + ord]

    def transpose(self):
        """
        Transpose a CSR matrix.

        Returns:
            CSR: the transpose of this matrix (or, equivalently, this matrix in CSC format).
        """

        return self._transpose(True)

    def transpose_coords(self):
        """
        Transpose a CSR matrix without retaining values.

        Returns:
            CSR: the transpose of this matrix (or, equivalently, this matrix in CSC format),
            without the value array.
        """

        return self._transpose(False)

    def _transpose(self, values):
        rowinds = np.empty(self.nnz, dtype=np.int32)
        for r in range(self.nrows):
            rsp = self.rowptrs[r]
            rep = self.rowptrs[r+1]
            rowinds[rsp:rep] = r

        align = np.empty(self.nnz, dtype=np.int32)
        colptrs = np.zeros(self.ncols + 1, dtype=np.int32)

        _csr_align(self.colinds, self.ncols, colptrs, align)

        n_rps = colptrs
        n_cis = rowinds[align].copy()
        if values and self.values is not None:
            n_vs = self.values[align].copy()
        else:
            n_vs = None

        return CSR(self.ncols, self.nrows, self.nnz, n_rps, n_cis, n_vs)


def csr_from_coo(rows, cols, vals, shape=None):
    """
    Create a CSR matrix from data in COO format.

    Args:
        rows(array-like): the row indices.
        cols(array-like): the column indices.
        vals(array-like): the data values; can be ``None``.
        shape(tuple): the array shape, or ``None`` to infer from row & column indices.
    """
    if shape is not None:
        nrows, ncols = shape
        assert np.max(rows) < nrows
        assert np.max(cols) < ncols
    else:
        nrows = np.max(rows) + 1
        ncols = np.max(cols) + 1

    nnz = len(rows)

    rowptrs = np.zeros(nrows + 1, dtype=np.int32)
    align = np.full(nnz, -1, dtype=np.int32)

    _csr_align(rows, nrows, rowptrs, align)

    colinds = cols[align].copy()
    values = vals[align].copy() if vals is not None else None

    return CSR(nrows, ncols, nnz, rowptrs, colinds, values)


CSR_type = CSR.class_type.instance_type


@njit
def _csr_align(rowinds, nrows, rowptrs, align):
    rcts = np.zeros(nrows, dtype=np.int32)
    for r in rowinds:
        rcts[r] += 1

    rowptrs[1:] = np.cumsum(rcts)
    rpos = rowptrs[:-1].copy()

    for i in range(len(rowinds)):
        row = rowinds[i]
        pos = rpos[row]
        align[pos] = i
        rpos[row] += 1


def csr_from_scipy(mat, copy=True):
    """
    Convert a scipy sparse matrix to an internal CSR.

    Args:
        mat(scipy.sparse.spmatrix): a SciPy sparse matrix.
        copy(bool): if ``False``, reuse the SciPy storage if possible.

    Returns:
        CSR: a CSR matrix.
    """
    if not sps.isspmatrix_csr(mat):
        mat = mat.tocsr(copy=copy)
    rp = np.require(mat.indptr, np.int32, 'C')
    if copy and rp is mat.indptr:
        rp = rp.copy()
    cs = np.require(mat.indices, np.int32, 'C')
    if copy and cs is mat.indices:
        cs = cs.copy()
    vs = mat.data.copy() if copy else mat.data
    return CSR(mat.shape[0], mat.shape[1], mat.nnz, rp, cs, vs)


def csr_to_scipy(mat):
    """
    Convert a CSR matrix to a SciPy :py:class:`scipy.sparse.csr_matrix`.

    Args:
        mat(CSR): A CSR matrix.

    Returns:
        scipy.sparse.csr_matrix:
            A SciPy sparse matrix with the same data.  It shares
            storage with ``matrix``.
    """
    if sps.isspmatrix(mat):
        warnings.warn('matrix already a SciPy matrix')
        return mat.tocsr()
    values = mat.values
    if values is None:
        values = np.full(mat.nnz, 1.0)
    return sps.csr_matrix((values, mat.colinds, mat.rowptrs), shape=(mat.nrows, mat.ncols))


def __scipy_csr_syrk(csr):
    sp = csr_to_scipy(csr)
    res = sp.T @ sp
    return csr_from_scipy(res, copy=False)


def __mkl_check_return(rv, call='<unknown>'):
    if rv:
        raise RuntimeError('MKL call {} failed with code {}'.format(call, rv))


_mkl_lib = None
__mkl_syrk_defs = '''
typedef void* sparse_matrix_t;

int mkl_sparse_d_create_csr(sparse_matrix_t *A, int indexing, int rows, int cols, int *rows_start, int *rows_end, int *col_indx, double *values);
int mkl_sparse_d_export_csr(const sparse_matrix_t source, int *indexing, int *rows, int *cols, int **rows_start, int **rows_end, int **col_indx, double **values);
int mkl_sparse_order(sparse_matrix_t A);
int mkl_sparse_destroy(sparse_matrix_t A);
int mkl_sparse_syrk (int operation, const sparse_matrix_t A, sparse_matrix_t *C);
'''


def __load_mkl():
    global _mkl_lib, _mkl_ffi
    if _mkl_lib is None:
        try:
            _logger.debug('trying to load MKL')
            import cffi
            _logger.debug('defining MKL FFI')
            _mkl_ffi = cffi.FFI()
            _mkl_ffi.cdef(__mkl_syrk_defs)
            _mkl_lib = _mkl_ffi.dlopen('mkl_rt')
            _logger.info('loaded MKL matrix operations')
        except (ImportError, OSError):
            _logger.info('failed to load cffi or MKL, falling back to scipy')
            _mkl_lib = False


def __mkl_csr_syrk(csr: CSR):
    sp = np.require(csr.rowptrs, np.intc, 'C')
    ep = np.require(csr.rowptrs[1:], np.intc, 'C')
    cols = np.require(csr.colinds, np.intc, 'C')
    vals = np.require(csr.values, np.float_, 'C')

    _logger.debug('syrk: processing %dx%d matrix (%d nnz)', csr.nrows, csr.ncols, csr.nnz)

    hdl = _mkl_ffi.new('sparse_matrix_t*')
    hdl2 = None
    _sp = _mkl_ffi.cast('int*', sp.ctypes.data)
    _ep = _mkl_ffi.cast('int*', ep.ctypes.data)
    _cols = _mkl_ffi.cast('int*', cols.ctypes.data)
    _vals = _mkl_ffi.cast('double*', vals.ctypes.data)
    rv = _mkl_lib.mkl_sparse_d_create_csr(hdl, 0, csr.nrows, csr.ncols, _sp, _ep, _cols, _vals)
    try:
        __mkl_check_return(rv, 'mkl_sparse_d_create_csr')

        _logger.debug('syrk: ordering matrix')
        rv = _mkl_lib.mkl_sparse_order(hdl[0])
        __mkl_check_return(rv, 'mkl_sparse_order')

        _logger.debug('syrk: multiplying matrix')
        hdl2 = _mkl_ffi.new('sparse_matrix_t*')
        rv = _mkl_lib.mkl_sparse_syrk(11, hdl[0], hdl2)
        __mkl_check_return(rv, 'mkl_sparse_syrk')
        _logger.debug('syrk: exporting matrix')

        indP = _mkl_ffi.new('int*')
        nrP = _mkl_ffi.new('int*')
        ncP = _mkl_ffi.new('int*')
        rsP = _mkl_ffi.new('int**')
        reP = _mkl_ffi.new('int**')
        ciP = _mkl_ffi.new('int**')
        vsP = _mkl_ffi.new('double**')
        rv = _mkl_lib.mkl_sparse_d_export_csr(hdl2[0], indP, nrP, ncP, rsP, reP, ciP, vsP)
        __mkl_check_return(rv, 'mkl_sparse_d_export_csr')
        if indP[0] != 0:
            raise ValueError('output index is not 0-indexed')
        nr = nrP[0]
        nc = ncP[0]
        assert nr == csr.ncols
        assert nc == csr.ncols
        rsB = _mkl_ffi.buffer(rsP[0], nr * _mkl_ffi.sizeof('int'))
        reB = _mkl_ffi.buffer(reP[0], nr * _mkl_ffi.sizeof('int'))
        rs = np.frombuffer(rsB, np.intc)
        re = np.frombuffer(reB, np.intc)
        assert np.all(rs[1:] == re[:nr-1])
        nnz = re[nr-1]
        _logger.debug('syrk: received %dx%d matrix (%d nnz)', nr, nc, nnz)
        _logger.debug('%s', rs)
        _logger.debug('%s', re)
        ciB = _mkl_ffi.buffer(ciP[0], nnz * _mkl_ffi.sizeof('int'))
        vsB = _mkl_ffi.buffer(vsP[0], nnz * _mkl_ffi.sizeof('double'))

        cols = np.frombuffer(ciB, np.intc)[:nnz].copy()
        _logger.debug('%s', cols)
        vals = np.frombuffer(vsB, np.float_)[:nnz].copy()
        _logger.debug('%s', vals)
        rowptrs = np.zeros(nr + 1, dtype=np.int32)
        rowptrs[1:] = re

        return CSR(nr, nc, nnz, rowptrs, cols, vals)

    finally:
        if hdl2 is not None:
            _logger.debug('syrk: freeing output matrix')
            _mkl_lib.mkl_sparse_destroy(hdl2[0])
            del hdl2

        _logger.debug('syrk: freeing input matrix')
        _mkl_lib.mkl_sparse_destroy(hdl[0])
        del hdl


def __triangular_to_symmetric(csr):
    # make sure it's square
    assert csr.nrows == csr.ncols
    # compute row indices
    rowinds = np.repeat(np.arange(csr.nrows), np.diff(csr.rowptrs))
    # how many extra values do we need?
    mask = rowinds < csr.colinds
    nvals = np.sum(mask)
    # make sure it's triangular
    assert nvals + np.sum(rowinds == csr.colinds) == csr.nnz
    _logger.debug('converting %dx%d matrix (%d nnz) to symmetric (%d non-diag vals)',
                  csr.nrows, csr.ncols, csr.nnz, nvals)

    # allocate new matrix space
    nnz2 = csr.nnz + nvals
    ri2 = np.empty(nnz2, np.int32)
    ri2[:csr.nnz] = rowinds
    ci2 = np.empty(nnz2, np.int32)
    ci2[:csr.nnz] = csr.colinds
    v2 = np.empty(nnz2, np.float_)
    v2[:csr.nnz] = csr.values

    # copy to do some COO work
    ri2[csr.nnz:] = csr.colinds[mask]
    ci2[csr.nnz:] = rowinds[mask]
    v2[csr.nnz:] = csr.values[mask]

    # and convert to CSR
    return csr_from_coo(ri2, ci2, v2, shape=(csr.nrows, csr.ncols))


def csr_syrk(csr):
    """
    The sparse SYRK (multiply transpose by matrix) operation. This returns
    :math:`M^T M` for an input matrix `M`.

    Args:
        csr(CSR): the input matrix :math:`M`.

    Returns:
        CSR: the matrix transpose multipled by the matrix.
    """
    __load_mkl()
    if _mkl_lib:
        if sps.isspmatrix(csr):
            csr = csr_from_scipy(csr)
        mat = __mkl_csr_syrk(csr)
        mat = __triangular_to_symmetric(mat)
        return mat
    else:
        return __scipy_csr_syrk(csr)


def sparse_ratings(ratings, scipy=False):
    """
    Convert a rating table to a sparse matrix of ratings.

    Args:
        ratings(pandas.DataFrame): a data table of (user, item, rating) triples.
        scipy: if ``True``, return a SciPy matrix instead of :py:class:`CSR`.

    Returns:
        RatingMatrix:
            a named tuple containing the sparse matrix, user index, and item index.
    """
    uidx = pd.Index(ratings.user.unique())
    iidx = pd.Index(ratings.item.unique())
    _logger.debug('creating matrix with %d ratings for %d items by %d users',
                  len(ratings), len(iidx), len(uidx))

    row_ind = uidx.get_indexer(ratings.user).astype(np.int32)
    col_ind = iidx.get_indexer(ratings.item).astype(np.int32)

    if 'rating' in ratings.columns:
        vals = np.require(ratings.rating.values, np.float64)
    else:
        vals = None

    matrix = csr_from_coo(row_ind, col_ind, vals, (len(uidx), len(iidx)))

    if scipy:
        matrix = csr_to_scipy(matrix)

    return RatingMatrix(matrix, uidx, iidx)
