"""
Efficient solver routines.
"""


import numpy as np

import cffi
import numba as n
from numba.extending import get_cython_function_address

__ffi = cffi.FFI()

__uplo_U = np.array([ord('U')], dtype=np.int8)
__uplo_L = np.array([ord('L')], dtype=np.int8)
__trans_N = np.array([ord('N')], dtype=np.int8)
__trans_T = np.array([ord('T')], dtype=np.int8)
__trans_C = np.array([ord('C')], dtype=np.int8)
__diag_U = np.array([ord('U')], dtype=np.int8)
__diag_N = np.array([ord('N')], dtype=np.int8)
__inc_1 = np.ones(1, dtype=np.int32)

__dtrsv = __ffi.cast("void (*) (char*, char*, char*, int*, double*, int*, double*, int*)",
                     get_cython_function_address("scipy.linalg.cython_blas", "dtrsv"))
__dposv = __ffi.cast("void (*) (char*, int*, int*, double*, int*, double*, int*, int*)",
                     get_cython_function_address("scipy.linalg.cython_lapack", "dposv"))


@n.njit
def _ref_sink(*args):
    return args


@n.njit(n.intc(n.float64[:, ::1], n.float64[::1], n.boolean))
def _dposv(A, b, lower):
    if A.shape[0] != A.shape[1]:
        return -11
    if A.shape[0] != b.shape[0]:
        return -12

    # dposv uses Fortran-layout arrays. Because we use C-layout arrays, we will
    # invert the meaning of 'lower' and 'trans', and the function will work fine.
    # We also need to swap index orders
    uplo = __uplo_U if lower else __uplo_L
    n_p = __ffi.from_buffer(np.array([A.shape[0]], dtype=np.intc))
    nrhs_p = __ffi.from_buffer(np.ones(1, dtype=np.intc))
    info = np.zeros(1, dtype=np.intc)
    info_p = __ffi.from_buffer(info)

    __dposv(__ffi.from_buffer(uplo), n_p, nrhs_p,
            __ffi.from_buffer(A), n_p,
            __ffi.from_buffer(b), n_p,
            info_p)

    _ref_sink(n_p, nrhs_p, info, info_p)

    return info[0]


def dposv(A, b, lower=False):
    """
    Interface to the BLAS dposv function.  A Numba-accessible verison without
    error checking is exposed as :py:func:`_dposv`.
    """
    info = _dposv(A, b, lower)
    if info < 0:
        raise ValueError('invalid args to dposv, code ' + str(info))
    elif info > 0:
        raise RuntimeError('error in dposv, code ' + str(info))
