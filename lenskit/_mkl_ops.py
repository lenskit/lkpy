import logging

import cffi

import numpy as np

from .matrix import CSR

_logger = logging.getLogger(__name__)

__mkl_syrk_defs = '''
typedef void* sparse_matrix_t;
struct matrix_descr {
    int  type;
    int  mode;
    int  diag;
};

int mkl_sparse_d_create_csr(sparse_matrix_t *A, int indexing, int rows, int cols,
                            int *rows_start, int *rows_end, int *col_indx, double *values);
int mkl_sparse_d_export_csr(const sparse_matrix_t source, int *indexing, int *rows, int *cols,
                            int **rows_start, int **rows_end, int **col_indx, double **values);
int mkl_sparse_order(sparse_matrix_t A);
int mkl_sparse_destroy(sparse_matrix_t A);

int mkl_sparse_syrk (int operation, const sparse_matrix_t A, sparse_matrix_t *C);
int mkl_sparse_d_mv (int operation, double alpha,
                     const sparse_matrix_t A, struct matrix_descr descr,
                     const double *x, double beta, double *y);
'''
_logger.debug('initializing CFFI interface')
_mkl_ffi = cffi.FFI()
_mkl_ffi.cdef(__mkl_syrk_defs)
try:
    _logger.debug('importing MKL')
    _mkl_lib = _mkl_ffi.dlopen('mkl_rt')
    _logger.info('Loaded MKL')
except OSError:
    _logger.info('Cannot load MKL')
    _mkl_lib = None


def _mkl_check_return(rv, call='<unknown>'):
    if rv:
        raise RuntimeError('MKL call {} failed with code {}'.format(call, rv))


def _mkl_basic_descr():
    desc = _mkl_ffi.new('struct matrix_descr*')
    desc.type = 20  # general matrix
    desc.mode = 0
    desc.diag = 0
    return desc


class SparseM:
    """
    Class encapsulating an MKL sparse matrix handle.
    """

    def __init__(self):
        self.h_ptr = _mkl_ffi.new('sparse_matrix_t*')

    @classmethod
    def from_csr(cls, csr):
        """
        Create an MKL sparse matrix from a LensKit CSR matrix.

        Args:
            csr(CSR): the input matrix.

        Returns:
            SparseM: a sparse matrix handle for the CSR matrix.
        """
        sp = np.require(csr.rowptrs, np.intc, 'C')
        ep = np.require(csr.rowptrs[1:], np.intc, 'C')
        cols = np.require(csr.colinds, np.intc, 'C')
        vals = np.require(csr.values, np.float_, 'C')

        m = SparseM()
        _logger.debug('creating MKL matrix 0x%08x from %dx%d CSR',
                      id(m), csr.nrows, csr.ncols)
        _sp = _mkl_ffi.cast('int*', sp.ctypes.data)
        _ep = _mkl_ffi.cast('int*', ep.ctypes.data)
        _cols = _mkl_ffi.cast('int*', cols.ctypes.data)
        _vals = _mkl_ffi.cast('double*', vals.ctypes.data)
        rv = _mkl_lib.mkl_sparse_d_create_csr(m.h_ptr, 0, csr.nrows, csr.ncols,
                                              _sp, _ep, _cols, _vals)
        _mkl_check_return(rv, 'mkl_sparse_d_create_csr')

        return m

    @property
    def handle(self):
        return self.h_ptr[0]

    def __del__(self):
        if self.h_ptr[0]:
            _logger.debug('destroying MKL sparse matrix 0x%08x', id(self))
            _mkl_lib.mkl_sparse_destroy(self.handle)

    def export(self):
        """
        Export an MKL sparse matrix as a LensKit CSR.

        Returns:
            CSR: the LensKit matrix.
        """
        indP = _mkl_ffi.new('int*')
        nrP = _mkl_ffi.new('int*')
        ncP = _mkl_ffi.new('int*')
        rsP = _mkl_ffi.new('int**')
        reP = _mkl_ffi.new('int**')
        ciP = _mkl_ffi.new('int**')
        vsP = _mkl_ffi.new('double**')
        rv = _mkl_lib.mkl_sparse_d_export_csr(self.handle, indP, nrP, ncP, rsP, reP, ciP, vsP)
        _mkl_check_return(rv, 'mkl_sparse_d_export_csr')
        if indP[0] != 0:
            raise ValueError('output index is not 0-indexed')
        nr = nrP[0]
        nc = ncP[0]
        reB = _mkl_ffi.buffer(reP[0], nr * _mkl_ffi.sizeof('int'))
        re = np.frombuffer(reB, np.intc)
        nnz = re[nr-1]
        ciB = _mkl_ffi.buffer(ciP[0], nnz * _mkl_ffi.sizeof('int'))
        vsB = _mkl_ffi.buffer(vsP[0], nnz * _mkl_ffi.sizeof('double'))

        cols = np.frombuffer(ciB, np.intc)[:nnz].copy()
        vals = np.frombuffer(vsB, np.float_)[:nnz].copy()
        rowptrs = np.zeros(nr + 1, dtype=np.int32)
        rowptrs[1:] = re

        return CSR(nr, nc, nnz, rowptrs, cols, vals)

    def mult_vec(self, alpha, x, beta, y):
        """
        Compute :math:`\\alpha A x + \\beta y`, where :math:`A` is this matrix.
        """
        desc = _mkl_basic_descr()
        x = np.require(x, np.float64, 'C')
        yout = np.require(y, np.float64, 'C')
        if yout is y:
            yout = yout.copy()

        _x = _mkl_ffi.cast('double*', x.ctypes.data)
        _y = _mkl_ffi.cast('double*', yout.ctypes.data)

        rv = _mkl_lib.mkl_sparse_d_mv(10, alpha, self.handle, desc[0], _x, beta, _y)
        _mkl_check_return(rv, 'mkl_sparse_d_mv')

        return yout


def csr_syrk(csr: CSR):
    """
    Interface to the ``mkl_sparse_syrk`` routine, with necessary setup and conversion.
    """

    _logger.debug('syrk: processing %dx%d matrix (%d nnz)', csr.nrows, csr.ncols, csr.nnz)

    src = SparseM.from_csr(csr)

    _logger.debug('syrk: ordering matrix')
    rv = _mkl_lib.mkl_sparse_order(src.handle)
    _mkl_check_return(rv, 'mkl_sparse_order')

    _logger.debug('syrk: multiplying matrix')
    mult = SparseM()
    rv = _mkl_lib.mkl_sparse_syrk(11, src.handle, mult.h_ptr)
    _mkl_check_return(rv, 'mkl_sparse_syrk')
    del src  # free a little memory

    _logger.debug('syrk: exporting matrix')
    result = mult.export()
    _logger.debug('syrk: received %dx%d matrix (%d nnz)',
                  result.nrows, result.ncols, result.nnz)
    return result
