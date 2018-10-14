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
    def __init__(self, shape, nnz, ptrs, inds, vals):
        self.nrows, self.ncols = shape
        self.nnz = nnz
        assert len(ptrs) == self.nrows + 1
        assert len(inds) >= nnz
        assert vals is None or len(vals) >= nnz
        self.rowptrs = ptrs
        self.colinds = inds
        self.values = vals


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
    else:
        nrows = np.max(rows) + 1
        ncols = np.max(cols) + 1

    nnz = len(rows)
    rptrs = np.zeros(nrows + 1, dtype=np.int32)
    cis = np.empty(nnz, dtype=np.int32)
    vs = np.empty(nnz) if vals is not None else None

    __csr_from_coo(rows, cols, vals, nrows, ncols, rptrs, cis, vs)

    return CSR((nrows, ncols), nnz, rptrs, cis, vs)


@njit('void(i4[:], i4[:], optional(double[:]), i4, i4, i4[:], i4[:], optional(double[:]))')
def __csr_from_coo(rows, cols, vals, nrows, ncols, rptrs, cis, vs):
    rcts = np.zeros(nrows, dtype=np.int32)
    for r in rows:
        rcts[r] += 1

    rptrs[1:] = np.cumsum(rcts)
    rpos = rptrs[:-1].copy()

    for i in range(len(cis)):
        row = rows[i]
        col = cols[i]
        pos = rpos[row]
        cis[pos] = col
        if vals is not None:
            assert vs is not None
            vs[pos] = vals[i]
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
    return CSR(mat.shape, mat.nnz, rp, cs, vs)


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
