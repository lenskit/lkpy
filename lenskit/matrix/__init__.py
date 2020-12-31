"""
Utilities for working with rating matrices.
"""

from collections import namedtuple
import logging

import pandas as pd
import numpy as np
import scipy.sparse as sps

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

__impl_mod = None


def _impl_mod():
    global __impl_mod
    if __impl_mod is None:
        from . import native
        __impl_mod = native
    return __impl_mod


def mkl_ops():
    """
    Import and return the MKL operations module.  This is only for internal use.
    """
    try:
        from . import _mkl_ops
        if _mkl_ops.clib:
            return _mkl_ops
        else:
            _logger.debug('MKL imported but not available')
            return None
    except ImportError as e:
        _logger.debug('MKL not importable: %s', e)
        return None


# This class lives here, un-JITted, so we avoid Numba compiles until necessary
class _CSR:
    """
    Internal implementation class for :py:class:`CSR`. If you work with CSRs from Numba,
    you will use a :func:`numba.jitclass`-ed version of this.

    Note that the ``values`` array is always present (unlike the Python shim), but is
    zero-length if no values are present.  This eases Numba type-checking.
    """
    def __init__(self, nrows, ncols, nnz, ptrs, inds, vals):
        self.nrows = nrows
        self.ncols = ncols
        self.nnz = nnz
        self.rowptrs = ptrs
        self.colinds = inds
        if vals is not None:
            self.values = vals
        else:
            self.values = np.zeros(0)

    def row(self, row):
        sp = self.rowptrs[row]
        ep = self.rowptrs[row + 1]

        v = np.zeros(self.ncols)
        cols = self.colinds[sp:ep]
        if self.values.size == 0:
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

        if self.values.size == 0:
            return np.full(ep - sp, 1.0)
        else:
            return self.values[sp:ep]

    def rowinds(self):
        ris = np.zeros(self.nnz, np.intc)
        for i in range(self.nrows):
            sp, ep = self.row_extent(i)
            ris[sp:ep] = i
        return ris


def _csr_delegate(name):
    def func(self):
        return getattr(self._N, name)

    return property(func)


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
    __slots__ = ['_N']

    def __init__(self, nrows=None, ncols=None, nnz=None, ptrs=None, inds=None, vals=None, N=None):
        if N is not None:
            self._N = N
        else:
            self._N = _CSR(nrows, ncols, nnz, ptrs, inds, vals)

    @classmethod
    def empty(cls, shape, row_nnzs, *, rpdtype=np.intc):
        """
        Create an empty CSR matrix.

        Args:
            shape(tuple): the array shape (rows,cols)
            row_nnzs(array-like): the number of nonzero entries for each row
        """
        nrows, ncols = shape
        assert len(row_nnzs) == nrows

        nnz = np.sum(row_nnzs, dtype=np.int64)
        rowptrs = np.zeros(nrows + 1, dtype=rpdtype)
        rowptrs[1:] = np.cumsum(row_nnzs, dtype=rpdtype)
        colinds = np.full(nnz, -1, dtype=np.intc)
        values = np.full(nnz, np.nan)

        return cls(nrows, ncols, nnz, rowptrs, colinds, values)

    @classmethod
    def from_coo(cls, rows, cols, vals, shape=None, rpdtype=np.intc):
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

        rowptrs = np.zeros(nrows + 1, dtype=rpdtype)
        align = np.full(nnz, -1, dtype=rpdtype)

        _impl_mod()._csr_align(rows, nrows, rowptrs, align)

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
        rp = np.require(mat.indptr, np.intc, 'C')
        if copy and rp is mat.indptr:
            rp = rp.copy()
        cs = np.require(mat.indices, np.intc, 'C')
        if copy and cs is mat.indices:
            cs = cs.copy()
        vs = mat.data.copy() if copy else mat.data
        return cls(mat.shape[0], mat.shape[1], mat.nnz, rp, cs, vs)

    def to_scipy(self):
        """
        Convert a CSR matrix to a SciPy :py:class:`scipy.sparse.csr_matrix`.  Avoids copying
        if possible.

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
    def N(self):
        """
        Get the native backing array.
        """
        # we do this *lazily*.
        n = self._N
        if isinstance(n, _CSR):
            # This is not yet upgraded
            impl = _impl_mod()
            if n.rowptrs.dtype == np.int64:
                self._N = impl._CSR64(n.nrows, n.ncols, n.nnz, n.rowptrs, n.colinds, n.values)
            else:
                self._N = impl._CSR(n.nrows, n.ncols, n.nnz, n.rowptrs, n.colinds, n.values)
            # make sure types behave how we expect
            assert impl.n.config.DISABLE_JIT or not isinstance(self._N, _CSR)
        return self._N

    nrows = _csr_delegate('nrows')
    ncols = _csr_delegate('ncols')
    nnz = _csr_delegate('nnz')
    rowptrs = _csr_delegate('rowptrs')
    colinds = _csr_delegate('colinds')

    @property
    def values(self):
        if self._N.values is not None and self._N.values.size:
            return self._N.values
        else:
            return None

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
            self._N.values = vs
        else:
            self._N.values = np.zeros(0)

    def subset_rows(self, begin, end):
        """
        Subset the rows in this matrix.
        """
        impl = _impl_mod()
        return CSR(N=impl._subset_rows(self.N, begin, end))

    def rowinds(self) -> np.ndarray:
        """
        Get the row indices from this array.  Combined with :py:attr:`colinds` and
        :py:attr:`values`, this can form a COO-format sparse matrix.
        """
        # we have to upgrade to Numba for this one - too slow in Python
        return self.N.rowinds()

    def row(self, row):
        """
        Return a row of this matrix as a dense ndarray.

        Args:
            row(int): the row index.

        Returns:
            numpy.ndarray: the row, with 0s in the place of missing values.
        """
        return self._N.row(row)

    def row_extent(self, row):
        """
        Get the extent of a row in the underlying column index and value arrays.

        Args:
            row(int): the row index.

        Returns:
            tuple: ``(s, e)``, where the row occupies positions :math:`[s, e)` in the
            CSR data.
        """
        return self._N.row_extent(row)

    def row_cs(self, row):
        """
        Get the column indcies for the stored values of a row.
        """
        return self._N.row_cs(row)

    def row_vs(self, row):
        """
        Get the stored values of a row.
        """
        return self._N.row_vs(row)

    def row_nnzs(self):
        """
        Get a vector of the number of nonzero entries in each row.

        .. note:: This method is not available from Numba.

        Returns:
            numpy.ndarray: the number of nonzero entries in each row.
        """
        return np.diff(self.rowptrs)

    def normalize_rows(self, normalization):
        """
        Normalize the rows of the matrix.

        .. note:: The normalization *ignores* missing values instead of treating
                  them as 0.

        .. note:: This method is not available from Numba.

        Args:
            normalization(str):
                The normalization to perform. Can be one of:

                * ``'center'`` - center rows about the mean
                * ``'unit'`` - convert rows to a unit vector

        Returns:
            numpy.ndarray:
                The normalization values for each row.
        """
        impl = _impl_mod()
        if normalization == 'center':
            return impl._center_rows(self.N)
        elif normalization == 'unit':
            return impl._unit_rows(self.N)
        else:
            raise ValueError('unknown normalization: ' + normalization)

    def transpose(self, values=True):
        """
        Transpose a CSR matrix.

        .. note:: This method is not available from Numba.

        Args:
            values(bool): whether to include the values in the transpose.

        Returns:
            CSR: the transpose of this matrix (or, equivalently, this matrix in CSC format).
        """
        impl = _impl_mod()
        n_rows = self.rowinds()
        rows = self.colinds
        n_vs = self.values if values else None
        if n_vs is not None:
            n_vs = n_vs.copy()

        rowptrs = impl._csr_align_inplace((self.ncols, self.nrows), rows, n_rows, n_vs)
        if self.rowptrs.dtype == np.int32:
            rowptrs = rowptrs.astype(np.int32)

        return CSR(self.ncols, self.nrows, self.nnz, rowptrs, n_rows, n_vs)

    def filter_nnzs(self, filt):
        """
        Filter the values along the full NNZ axis.

        Args:
            filt(ndarray):
                a logical array of length :attr:`nnz` that indicates the values to keep.

        Returns:
            CSR: The filtered sparse matrix.
        """
        if len(filt) != self.nnz:
            raise ValueError('filter has length %d, expected %d' % (len(filt), self.nnz))
        rps2 = np.zeros_like(self.rowptrs)
        for i in range(self.nrows):
            sp, ep = self.row_extent(i)
            rlen = np.sum(filt[sp:ep])
            rps2[i+1] = rps2[i] + rlen

        nnz2 = rps2[-1]
        assert nnz2 == np.sum(filt)

        cis2 = self.colinds[filt]
        vs = self.values
        vs2 = None if vs is None else vs[filt]

        return CSR(self.nrows, self.ncols, nnz2, rps2, cis2, vs2)

    def __str__(self):
        return '<CSR {}x{} ({} nnz)>'.format(self.nrows, self.ncols, self.nnz)

    def __repr__(self):
        repr = '<CSR {}x{} ({} nnz)'.format(self.nrows, self.ncols, self.nnz)
        repr += ' {\n'
        repr += '  rowptrs={}\n'.format(self.rowptrs)
        repr += '  colinds={}\n'.format(self.colinds)
        repr += '  values={}\n'.format(self.values)
        repr += '}'
        return repr

    def __getstate__(self):
        return dict(shape=(self.nrows, self.ncols), nnz=self.nnz,
                    rowptrs=self.rowptrs, colinds=self.colinds, values=self.values)

    def __setstate__(self, state):
        nrows, ncols = state['shape']
        nnz = state['nnz']
        rps = state['rowptrs']
        cis = state['colinds']
        vs = state['values']
        self._N = _CSR(nrows, ncols, nnz, rps, cis, vs)


def sparse_ratings(ratings, scipy=False, *, users=None, items=None):
    """
    Convert a rating table to a sparse matrix of ratings.

    Args:
        ratings(pandas.DataFrame): a data table of (user, item, rating) triples.
        scipy: if ``True`` or ``'csr'``, return a SciPy csr matrix instead of
        :py:class:`CSR`. if ``'coo'``, return a SciPy coo matrix.
        users(pandas.Index): an index of user IDs.
        items(pandas.Index): an index of items IDs.

    Returns:
        RatingMatrix:
            a named tuple containing the sparse matrix, user index, and item index.
    """
    if users is None:
        users = pd.Index(np.unique(ratings.user), name='user')

    if items is None:
        items = pd.Index(np.unique(ratings.item), name='item')

    _logger.debug('creating matrix with %d ratings for %d items by %d users',
                  len(ratings), len(items), len(users))

    row_ind = users.get_indexer(ratings.user).astype(np.intc)
    if np.any(row_ind < 0):
        raise ValueError('provided user index does not cover all users')
    col_ind = items.get_indexer(ratings.item).astype(np.intc)
    if np.any(col_ind < 0):
        raise ValueError('provided item index does not cover all users')

    if 'rating' in ratings.columns:
        vals = np.require(ratings.rating.values, np.float64)
    else:
        vals = None

    if scipy == 'coo':
        matrix = sps.coo_matrix(
            (vals, (row_ind, col_ind)), shape=(len(users), len(items))
        )
    else:
        matrix = CSR.from_coo(row_ind, col_ind, vals, (len(users), len(items)))
        if scipy:
            matrix = matrix.to_scipy()

    return RatingMatrix(matrix, users, items)
