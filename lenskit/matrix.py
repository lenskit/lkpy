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
from numba import njit, jitclass, prange

_logger = logging.getLogger(__name__)

RatingMatrix = namedtuple('RatingMatrix', ['matrix', 'users', 'items'])
RatingMatrix.__doc__ = """
A rating matrix with associated indices.

Attributes:
    matrix(CSR or scipy.sparse.csr_matrix):
        The rating matrix, with users on rows and items on columns.
    users(pandas.Index): mapping from user IDs to row numbers.
    items(pandas.Index): mapping from item IDs to column numbers.
"""


def mkl_ops():
    """
    Import and return the MKL operations module.  This is only for internal use.
    """
    try:
        from . import _mkl_ops
        if _mkl_ops._mkl_lib:
            return _mkl_ops
        else:
            return None
    except ImportError:
        return None


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

    def row_extent(self, row):
        sp = self.rowptrs[row]
        ep = self.rowptrs[row+1]
        return (sp, ep)

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
        _csr_sort(self.nrows, self.rowptrs, self.colinds, self.values)

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


@njit(n.void(n.intc, n.int32[:], n.int32[:], n.double[:]),
      parallel=True, nogil=True)
def _csr_sort(nrows, rowptrs, colinds, values):
    assert len(rowptrs) > nrows
    for i in prange(nrows):
        sp = rowptrs[i]
        ep = rowptrs[i+1]
        if ep > sp:
            ord = np.argsort(values[sp:ep])
            ord = ord[::-1]
            colinds[sp:ep] = colinds[sp + ord]
            values[sp:ep] = values[sp + ord]


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
    assert len(cols) == nnz
    assert vals is None or len(vals) == nnz

    rowptrs = np.zeros(nrows + 1, dtype=np.int32)
    align = np.full(nnz, -1, dtype=np.int32)

    _csr_align(rows, nrows, rowptrs, align)

    colinds = cols[align].copy()
    values = vals[align].copy() if vals is not None else None

    return CSR(nrows, ncols, nnz, rowptrs, colinds, values)


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


def csr_rowinds(csr):
    """
    Get the row indices for a CSR matrix.

    Args:
        csr(CSR): a CSR matrix.

    Returns:
        np.ndarray: the row index array for the CSR matrix.
    """
    return np.repeat(np.arange(csr.nrows), np.diff(csr.rowptrs))


def csr_save(csr: CSR, prefix=None):
    """
    Extract data needed to save a CSR matrix.  This is intended to be used with, for
    example, :py:func:`numpy.savez` to save a matrix::

        np.savez_compressed('file.npz', **csr_save(csr))

    The ``prefix`` allows multiple matrices to be saved in a single file::

        data = {}
        data.update(csr_save(m1, prefix='m1'))
        data.update(csr_save(m2, prefix='m2'))
        np.savez_compressed('file.npz', **data)

    Args:
        csr(CSR): the matrix to save.
        prefix(str): the prefix for the data keys.

    Returns:
        dict: a dictionary of data to save the matrix.
    """
    if prefix is None:
        prefix = ''
    return {
        prefix + 'ncols': csr.ncols,
        prefix + 'nrows': csr.nrows,
        prefix + 'rowptrs': csr.rowptrs,
        prefix + 'colinds': csr.colinds,
        prefix + 'values': csr.values
    }


def csr_load(data, prefix=None):
    """
    Rematerialize a CSR matrix from loaded data.  The inverse of :py:func:`csr_save`.

    Args:
        data(dict-like): the input data.
        prefix(str): the prefix for the data keys.

    Returns:
        CSR: the matrix described by ``data``.
    """
    if prefix is None:
        prefix = ''
    ncols = int(data[prefix + 'ncols'])
    nrows = int(data[prefix + 'nrows'])
    rowptrs = data[prefix + 'rowptrs']
    colinds = data[prefix + 'colinds']
    values = data[prefix + 'values']
    if values.ndim == 0:
        values = None
    return CSR(nrows, ncols, len(colinds), rowptrs, colinds, values)


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
    uidx = pd.Index(ratings.user.unique(), name='user')
    iidx = pd.Index(ratings.item.unique(), name='item')
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
