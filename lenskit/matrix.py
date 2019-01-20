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
class _CSR:
    """
    Internal implementation class for :py:class:`CSR`. If you work with CSRs from Numba,
    you will use this.
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


class CSR:
    """
    Simple compressed sparse row matrix.  This is like :py:class:`scipy.sparse.csr_matrix`, with
    a couple of useful differences:

    * It is backed by a Numba jitclass, so it can be directly used from Numba-optimized functions.
    * The value array is optional, for cases in which only the matrix structure is required.
    * The value array, if present, is always double-precision.

    You generally don't want to create this class yourself with the constructor.  Instead, use one
    of its class methods.

    If you need to pass an instance off to a Numba-compiled function, use :py:attr:`N`::

        _some_numba_fun(csr.N)

    We use the indirection between this and the Numba jitclass so that the main CSR implementation
    can be pickled, and so that we can have class and instance methods that are not compatible with
    jitclass but which are useful from interpreted code.

    Attributes:
        N(_CSR): the Numba jitclass backing (has the same attributes and most methods).
        nrows(int): the number of rows.
        ncols(int): the number of columns.
        nnz(int): the number of entries.
        rowptrs(numpy.ndarray): the row pointers.
        colinds(numpy.ndarray): the column indices.
        values(numpy.ndarray): the values
    """
    __slots__ = ['N']

    def __init__(self, nrows=None, ncols=None, nnz=None, ptrs=None, inds=None, vals=None, N=None):
        if N is not None:
            self.N = N
        else:
            self.N = _CSR(nrows, ncols, nnz, ptrs, inds, vals)

    @classmethod
    def from_coo(cls, rows, cols, vals, shape=None):
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

        cols = cols[align].copy()
        vals = vals[align].copy() if vals is not None else None

        return cls(nrows, ncols, nnz, rowptrs, cols, vals)

    @classmethod
    def from_scipy(cls, mat, copy=True):
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
        return cls(mat.shape[0], mat.shape[1], mat.nnz, rp, cs, vs)

    def to_scipy(self):
        """
        Convert a CSR matrix to a SciPy :py:class:`scipy.sparse.csr_matrix`.

        Args:
            self(CSR): A CSR matrix.

        Returns:
            scipy.sparse.csr_matrix:
                A SciPy sparse matrix with the same data.
        """
        values = self.values
        if values is None:
            values = np.full(self.nnz, 1.0)
        return sps.csr_matrix((values, self.colinds, self.rowptrs), shape=(self.nrows, self.ncols))

    @property
    def nrows(self) -> int:
        return self.N.nrows

    @property
    def ncols(self) -> int:
        return self.N.ncols

    @property
    def nnz(self) -> int:
        return self.N.nnz

    @property
    def rowptrs(self) -> np.ndarray:
        return self.N.rowptrs

    @property
    def colinds(self) -> np.ndarray:
        return self.N.colinds

    @property
    def values(self) -> np.ndarray:
        return self.N.values

    @values.setter
    def values(self, vs: np.ndarray):
        if vs is not None:
            if not isinstance(vs, np.ndarray):
                raise TypeError('values not an ndarray')
            if vs.ndim != 1:
                raise ValueError('values has {} dimensions, expected 1'.format(vs.ndims))
            if vs.shape[0] < self.nnz:
                s = 'values has only {} entries (expected at least {})'
                raise ValueError(s.format(vs.shape[0], self.nnz))

            vs = vs[:self.nnz]
            vs = np.require(vs, 'f8')

        self.N.values = vs

    def rowinds(self) -> np.ndarray:
        """
        Get the row indices from this array.  Combined with :py:attr:`colinds` and
        :py:attr:`values`, this can form a COO-format sparse matrix.

        .. note:: This method is not available from Numba.
        """
        return np.repeat(np.arange(self.nrows, dtype=np.int32), np.diff(self.rowptrs))

    def row(self, row):
        """
        Return a row of this matrix as a dense ndarray.

        Args:
            row(int): the row index.

        Returns:
            numpy.ndarray: the row, with 0s in the place of missing values.
        """
        return self.N.row(row)

    def row_extent(self, row):
        """
        Get the extent of a row in the underlying column index and value arrays.

        Args:
            row(int): the row index.

        Returns:
            tuple: ``(s, e)``, where the row occupies positions :math:`[s, e)` in the
            CSR data.
        """
        return self.N.row_extent(row)

    def row_cs(self, row):
        """
        Get the column indcies for the stored values of a row.
        """
        return self.N.row_cs(row)

    def row_vs(self, row):
        """
        Get the stored values of a row.
        """
        return self.N.row_vs(row)

    def row_nnzs(self):
        """
        Get a vector of the number of nonzero entries in each row.

        .. note:: This method is not available from Numba.

        Returns:
            numpy.ndarray: the number of nonzero entries in each row.
        """
        return np.diff(self.rowptrs)

    def sort_values(self):
        """
        Sort CSR rows in nonincreasing order by value.

        .. note:: This method is not available from Numba.
        """
        _csr_sort(self.nrows, self.rowptrs, self.colinds, self.values)

    def transpose(self, values=True):
        """
        Transpose a CSR matrix.

        .. note:: This method is not available from Numba.

        Args:
            values(bool): whether to include the values in the transpose.

        Returns:
            CSR: the transpose of this matrix (or, equivalently, this matrix in CSC format).
        """

        rowinds = self.rowinds()
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

    def __str__(self):
        return '<CSR {}x{} ({} nnz)>'.format(self.nrows, self.ncols, self.nnz)

    def __getstate__(self):
        return dict(shape=(self.nrows, self.ncols), nnz=self.nnz,
                    rowptrs=self.rowptrs, colinds=self.colinds, values=self.values)

    def __setstate__(self, state):
        nrows, ncols = state['shape']
        nnz = state['nnz']
        rps = state['rowptrs']
        cis = state['colinds']
        vs = state['values']
        self.N = _CSR(nrows, ncols, nnz, rps, cis, vs)


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

    matrix = CSR.from_coo(row_ind, col_ind, vals, (len(uidx), len(iidx)))

    if scipy:
        matrix = matrix.to_scipy()

    return RatingMatrix(matrix, uidx, iidx)
