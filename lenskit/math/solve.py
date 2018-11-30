"""
Efficient solver routines.
"""


import numpy as np

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

__dtrsv = __ffi.cast("void (*) (char*, char*, char*, int*, double*, int*, double*, int*)",
                     get_cython_function_address("scipy.linalg.cython_blas", "dtrsv"))


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


def solve_tri(A, b, transpose=False, lower=True):
    """
    Solve the system :math:`Ax = b`, where :math:`A` is triangular.
    This is equivalent to :py:fun:`scipy.linalg.solve_triangular`, but does *not*
    check for non-singularity.  It is a thin wrapper around the BLAS ``dtrsv``
    function.

    Args:
        A(ndarray): the matrix.
        b(ndarray): the taget vector.
        transpose(bool): whether to solve :math:`Ax = b` or :math:`A^T x = b`.
        lower(bool): whether :math:`A` is lower- or upper-triangular.
    """
    x = b.copy()
    _dtrsv(lower, transpose, A, x)
    return x
