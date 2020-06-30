import os
import pathlib
import logging

import cffi
from distutils import ccompiler
from numba import njit, types as nt

try:
    from numba.core.typing import cffi_utils
except ImportError:
    # earlier Numba versions have an old module
    from numba import cffi_support as cffi_utils

import numpy as np

from . import CSR, _impl_mod
_CSR = _impl_mod()._CSR

_logger = logging.getLogger(__name__)
__dir = pathlib.Path(__file__).parent

__cc = ccompiler.new_compiler()
_mkl_so = __dir / __cc.shared_object_filename('mkl_ops')
__mkl_defs = (__dir / 'mkl_ops.h').read_text()
ffi = cffi.FFI()
ffi.cdef(__mkl_defs.replace('EXPORT ', ''))
try:
    clib = ffi.dlopen(os.fspath(_mkl_so))

    _lk_mkl_spcreate = clib.lk_mkl_spcreate
    _lk_mkl_spsubset = clib.lk_mkl_spsubset
    _lk_mkl_spfree = clib.lk_mkl_spfree
    _lk_mkl_sporder = clib.lk_mkl_sporder
    _lk_mkl_spopt = clib.lk_mkl_spopt
    _lk_mkl_spmv = clib.lk_mkl_spmv
    _lk_mkl_spmab = clib.lk_mkl_spmab
    _lk_mkl_spmabt = clib.lk_mkl_spmabt
    _lk_mkl_spexport = clib.lk_mkl_spexport

    # silly pointer interface
    _lk_mkl_spexport_p = clib.lk_mkl_spexport_p
    _lk_mkl_spe_free = clib.lk_mkl_spe_free
    _lk_mkl_spe_nrows = clib.lk_mkl_spe_nrows
    _lk_mkl_spe_ncols = clib.lk_mkl_spe_ncols
    _lk_mkl_spe_row_sp = clib.lk_mkl_spe_row_sp
    _lk_mkl_spe_row_ep = clib.lk_mkl_spe_row_ep
    _lk_mkl_spe_colinds = clib.lk_mkl_spe_colinds
    _lk_mkl_spe_values = clib.lk_mkl_spe_values
except OSError:
    clib = None

# support intptr_t
if cffi_utils is not None:
    # if we don't have working Numba, skip this
    cffi_utils.register_type(ffi.typeof('intptr_t'), nt.intp)

# extract sizes
_int_size = ffi.sizeof('int')
_dbl_size = ffi.sizeof('double')

_mkl_errors = [
    'SPARSE_STATUS_SUCCESS',
    'SPARSE_STATUS_NOT_INITIALIZED',
    'SPARSE_STATUS_ALLOC_FAILED',
    'SPARSE_STATUS_INVALID_VALUE',
    'SPARSE_STATUS_EXECUTION_FAILED',
    'SPARSE_STATUS_INTERNAL_ERROR',
    'SPARSE_STATUS_NOT_SUPPORTED'
]


def _mkl_check_return(rv, call='<unknown>'):
    if rv:
        if rv >= 0 and rv < len(_mkl_errors):
            desc = _mkl_errors[rv]
        else:
            desc = 'unknown'
        raise RuntimeError('MKL call {} failed with code {} ({})'.format(call, rv, desc))


class SparseM:
    """
    Class encapsulating an MKL sparse matrix handle.
    """

    def __init__(self):
        self.ptr = None

    @classmethod
    def from_csr(cls, csr):
        """
        Create an MKL sparse matrix from a LensKit CSR matrix.

        Args:
            csr(CSR): the input matrix.

        Returns:
            SparseM: a sparse matrix handle for the CSR matrix.
        """
        m = SparseM()
        m.ptr = _from_csr(csr.N)
        if not m.ptr:
            raise RuntimeError('MKL matrix creation failed')

        m._csr = csr  # save the CSR matrix to ensure it oulives the SparseH
        return m

    def __del__(self):
        if self.ptr:
            _logger.debug('destroying MKL sparse matrix 0x%08x', id(self))
            clib.lk_mkl_spfree(self.ptr)

    def export(self):
        """
        Export an MKL sparse matrix as a LensKit CSR.

        Returns:
            CSR: the LensKit matrix.
        """
        csr = _to_csr(self.ptr)
        if not csr:
            raise RuntimeError('MKL failed to export CSR')

        return CSR(N=csr)

    def mult_vec(self, alpha, x, beta, y):
        """
        Compute :math:`\\alpha A x + \\beta y`, where :math:`A` is this matrix.
        """
        x = np.require(x, np.float64, 'C')
        yout = np.require(y, np.float64, 'C')
        if yout is y:
            yout = yout.copy()

        _x = ffi.from_buffer('double[]', x)
        _y = ffi.from_buffer('double[]', yout)

        rv = clib.lk_mkl_spmv(alpha, self.ptr, _x, beta, _y)
        _mkl_check_return(rv, 'mkl_sparse_d_mv')

        return yout


@njit
def _from_csr(csr: _CSR):
    """
    Convert a Numba CSR to an MKL sparse matrix handle.
    """
    _sp = ffi.from_buffer(csr.rowptrs)
    _cols = ffi.from_buffer(csr.colinds)
    _vals = ffi.from_buffer(csr.values)
    return _lk_mkl_spcreate(csr.nrows, csr.ncols, _sp, _cols, _vals)


@njit
def _from_csr_ss(csr: _CSR, rsp, rep):
    """
    Convert a subset of a Numba CSR to an MKL sparse matrix handle.
    """
    _sp = ffi.from_buffer(csr.rowptrs)
    _cols = ffi.from_buffer(csr.colinds)
    _vals = ffi.from_buffer(csr.values)
    return _lk_mkl_spsubset(rsp, rep, csr.ncols, _sp, _cols, _vals)


@njit
def _to_csr(smh):
    """
    Convert an MKL sparse matrix handle to a Numba CSR.
    """
    rvp = _lk_mkl_spexport_p(smh)
    if rvp is None:
        return None

    nrows = _lk_mkl_spe_nrows(rvp)
    ncols = _lk_mkl_spe_ncols(rvp)

    sp = _lk_mkl_spe_row_sp(rvp)
    ep = _lk_mkl_spe_row_ep(rvp)
    cis = _lk_mkl_spe_colinds(rvp)
    vs = _lk_mkl_spe_values(rvp)

    rowptrs = np.zeros(nrows + 1, dtype=np.intc)
    nnz = 0
    for i in range(nrows):
        nnz += ep[i] - sp[i]
        rowptrs[i+1] = nnz

    colinds = np.zeros(nnz, dtype=np.intc)
    values = np.zeros(nnz)

    for i in range(nrows):
        rs = rowptrs[i]
        re = rowptrs[i+1]
        ss = sp[i]
        for j in range(re - rs):
            colinds[rs + j] = cis[ss + j]
            values[rs + j] = vs[ss + j]

    _lk_mkl_spe_free(rvp)

    return _CSR(nrows, ncols, nnz, rowptrs, colinds, values)
