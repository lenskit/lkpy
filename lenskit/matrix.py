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


def csr_from_coo(rows, cols, vals, shape=None):
    """
    Create a CSR matrix from data in COO format.

    Args:
        rows(array-like): the row indices.
        cols(array-like): the column indices.
        vals(array-like): the data values
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

    __csr_align(rows, nrows, rowptrs, align)

    colinds = cols[align].copy()
    values = vals[align].copy() if vals is not None else None

    return CSR(nrows, ncols, nnz, rowptrs, colinds, values)


@njit
def __csr_align(rowinds, nrows, rowptrs, align):
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


def sparse_ratings(ratings, layout='csr'):
    """
    Convert a rating table to a sparse matrix of ratings.

    Args:
        ratings(pandas.DataFrame): a data table of (user, item, rating) triples.
        layout: the sparse matrix layout to use

    Returns:
        scipy.sparse.spmatrix:
            a sparse matrix with users on the rows and items on the columns.
    """
    if layout not in ('csr', 'csc', 'coo'):
        raise ValueError('invalid matrix layout ' + layout)

    uidx = pd.Index(ratings.user.unique())
    iidx = pd.Index(ratings.item.unique())
    _logger.debug('creating matrix with %d ratings for %d items by %d users',
                  len(ratings), len(iidx), len(uidx))

    row_ind = uidx.get_indexer(ratings.user)
    col_ind = iidx.get_indexer(ratings.item)

    mkmat = getattr(sps, layout + '_matrix')
    if 'rating' in ratings.columns:
        vals = ratings.rating.values
    else:
        vals = np.full(len(ratings), 1.0)
    matrix = mkmat((vals, (row_ind, col_ind)),
                   shape=(len(uidx), len(iidx)))

    return RatingMatrix(matrix, uidx, iidx)
