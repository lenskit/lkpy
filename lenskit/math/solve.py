"""
Efficient solver routines.
"""


import numpy as np

import ctypes
import cffi
import numba as n
from numba.extending import get_cython_function_address

__ffi = cffi.FFI()

__uplo_U = np.array(ord('U'), dtype=np.int8)
__uplo_L = np.array(ord('L'), dtype=np.int8)
__trans_N = np.array(ord('N'), dtype=np.int8)
__trans_T = np.array(ord('T'), dtype=np.int8)
__trans_C = np.array(ord('C'), dtype=np.int8)
__diag_U = np.array(ord('U'), dtype=np.int8)
__diag_N = np.array(ord('N'), dtype=np.int8)
__inc_1 = np.ones(1, dtype=np.int32)

__dtrsv_addr = get_cython_function_address("scipy.linalg.cython_blas", "dtrsv")
__c_p = ctypes.c_char_p
__i_p = ctypes.POINTER(ctypes.c_int)
__d_p = ctypes.POINTER(ctypes.c_double)
__dtrsv_type = ctypes.CFUNCTYPE(None, __c_p, __c_p, __c_p, __i_p, __d_p, __i_p, __d_p, __i_p)
# __dtrsv = __dtrsv_type(__dtrsv_addr)
__dtrsv = __ffi.cast("void (*) (char*, char*, char*, int*, double*, int*, double*, int*)",
                     get_cython_function_address("scipy.linalg.cython_blas", "dtrsv"))

__dtrtrs_addr = get_cython_function_address("scipy.linalg.cython_lapack", "dtrtrs")
__c_p = ctypes.c_char_p
__i_p = ctypes.POINTER(ctypes.c_int)
__d_p = ctypes.POINTER(ctypes.c_double)
__dtrtrs_type = ctypes.CFUNCTYPE(None, __c_p, __c_p, __c_p, __i_p, __i_p, __d_p, __i_p, __d_p, __i_p, __i_p)
__dtrtrs = __dtrtrs_type(__dtrtrs_addr)


@n.njit(n.void(n.boolean, n.boolean, n.double[:, ::1], n.double[::1]), nogil=True)
def _dtrsv(lower, trans, a, x):
    inc1 = __ffi.from_buffer(__inc_1)

    # dtrsv uses Fortran-layout arrays. Because we use C-layout arrays, we will
    # invert the meaning of 'lower' and 'trans', and the function will work fine.
    # We also need to swap index orders
    uplo = __uplo_U if lower else __uplo_L
    tspec = __trans_N if trans else __trans_T

    n_p = np.array([a.shape[0]], dtype=np.intc)
    n_p = __ffi.from_buffer(n_p)
    lda_p = np.array([a.shape[1]], dtype=np.intc)
    lda_p = __ffi.from_buffer(lda_p)

    __dtrsv(__ffi.from_buffer(uplo), __ffi.from_buffer(tspec), __ffi.from_buffer(__diag_N),
            n_p, __ffi.from_buffer(a), lda_p,
            __ffi.from_buffer(x), inc1)


@n.njit(nogil=True)
def solve_ltri(A, b, transpose=False):
    """
    Solve the system :math:`Ax = b`, where :math:`A` is lower-triangular.
    This is equivalent to :py:fun:`scipy.linalg.solve_triangular`.

    Args:
        A(ndarray): the matrix.
        b(ndarray): the taget vector.
    """
    x = b.copy()
    _dtrsv(True, transpose, A, x)
    return x


@n.njit(nogil=True)
def solve_utri(A, b, transpose=False):
    """
    Solve the system :math:`Ax = b`, where :math:`A` is upper-triangular.
    This is equivalent to :py:fun:`scipy.linalg.solve_triangular`.

    Args:
        A(ndarray): the matrix.
        b(ndarray): the taget vector.
    """
    x = b.copy()
    _dtrsv(False, transpose, A, x)
    return x
