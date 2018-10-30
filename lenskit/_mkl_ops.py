import logging

import cffi

import numpy as np

from .matrix import CSR

_logger = logging.getLogger(__name__)

__mkl_syrk_defs = '''
typedef void* sparse_matrix_t;

int mkl_sparse_d_create_csr(sparse_matrix_t *A, int indexing, int rows, int cols,
                            int *rows_start, int *rows_end, int *col_indx, double *values);
int mkl_sparse_d_export_csr(const sparse_matrix_t source, int *indexing, int *rows, int *cols,
                            int **rows_start, int **rows_end, int **col_indx, double **values);
int mkl_sparse_order(sparse_matrix_t A);
int mkl_sparse_destroy(sparse_matrix_t A);
int mkl_sparse_syrk (int operation, const sparse_matrix_t A, sparse_matrix_t *C);
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


class _MKL_SparseH:
    """
    Class encapsulating an MKL sparse matrix handle.
    """

    def __init__(self):
        self.h_ptr = _mkl_ffi.new('sparse_matrix_t*')

    @property
    def handle(self):
        return self.h_ptr[0]

    def __del__(self):
        if self.h_ptr[0]:
            _mkl_lib.mkl_sparse_destroy(self.handle)

    def export(self):
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


def csr_syrk(csr: CSR):
    """
    Interface to the ``mkl_sparse_syrk`` routine, with necessary setup and conversion.
    """
    sp = np.require(csr.rowptrs, np.intc, 'C')
    ep = np.require(csr.rowptrs[1:], np.intc, 'C')
    cols = np.require(csr.colinds, np.intc, 'C')
    vals = np.require(csr.values, np.float_, 'C')

    _logger.debug('syrk: processing %dx%d matrix (%d nnz)', csr.nrows, csr.ncols, csr.nnz)

    src = _MKL_SparseH()
    _sp = _mkl_ffi.cast('int*', sp.ctypes.data)
    _ep = _mkl_ffi.cast('int*', ep.ctypes.data)
    _cols = _mkl_ffi.cast('int*', cols.ctypes.data)
    _vals = _mkl_ffi.cast('double*', vals.ctypes.data)
    rv = _mkl_lib.mkl_sparse_d_create_csr(src.h_ptr, 0, csr.nrows, csr.ncols,
                                          _sp, _ep, _cols, _vals)
    _mkl_check_return(rv, 'mkl_sparse_d_create_csr')

    _logger.debug('syrk: ordering matrix')
    rv = _mkl_lib.mkl_sparse_order(src.handle)
    _mkl_check_return(rv, 'mkl_sparse_order')

    _logger.debug('syrk: multiplying matrix')
    mult = _MKL_SparseH()
    rv = _mkl_lib.mkl_sparse_syrk(11, src.handle, mult.h_ptr)
    _mkl_check_return(rv, 'mkl_sparse_syrk')
    _logger.debug('syrk: exporting matrix')

    result = mult.export()
    _logger.debug('syrk: received %dx%d matrix (%d nnz)',
                  result.nrows, result.ncols, result.nnz)
    return result
