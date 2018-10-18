"""
Utilities for working with rating matrices.
"""

from collections import namedtuple
import logging

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
        rowptrs(array-like): the row pointers.
        colinds(array-like): the column indices.
        values(array-like): the values
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
    rp = mat.indptr.copy() if copy else mat.indptr
    cs = mat.indices.copy() if copy else mat.indices
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
    values = mat.values
    if values is None:
        values = np.full(mat.nnz, 1.0)
    return sps.csr_matrix((values, mat.colinds, mat.rowptrs), shape=(mat.nrows, mat.ncols))


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
        vals = ratings.rating.values
        if vals.dtype != np.float_:
            vals = vals.astype('f8')
    else:
        vals = None

    matrix = csr_from_coo(row_ind, col_ind, vals, (len(uidx), len(iidx)))

    if scipy:
        matrix = csr_to_scipy(matrix)

    return RatingMatrix(matrix, uidx, iidx)
