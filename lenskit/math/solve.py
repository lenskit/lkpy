"""
Efficient solver routines.
"""

from .blas import _dtrsv, _dposv


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


def dposv(A, b, lower=False):
    """
    Solve a symmetric positive system in-place.  This is a wrapper around the
    BLAS ``dposv`` function.
    """
    info = _dposv(A, b, lower)
    if info < 0:
        raise ValueError('invalid args to dposv, code ' + str(info))
    elif info > 0:
        raise RuntimeError('error in dposv, code ' + str(info))
