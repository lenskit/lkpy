"""
Utility functions for precondition checks.
"""

import warnings


def _get_size(m, d):
    if d is None:
        return len(m)
    else:
        return m.shape[d]


def check_value(expr, msg, *args, warn=False):
    if not expr:
        if warn:
            warnings.warn(msg.format(*args))
        else:
            raise ValueError(msg.format(*args))


def check_dimension(m1, m2, msg=None, d1=None, d2=None):
    """
    Check the dimensions of a pair of matrices or arrays.

    Args:
        m1(array-like): the left matrix or array
        m2(array-like): the right matrix or array
        d1(int):
            the left dimension to check.  If an integer, then this method will
            check ``m1.shape[d1]``; if ``None``, then ``len(m1)``.
        d2(int):
            the right dimension to check.  If an integer, then this method will
            check ``m2.shape[d2]``; if ``None``, then ``len(m2)``.
    """
    sz1 = _get_size(m1, d1)
    sz2 = _get_size(m2, d2)

    if sz1 != sz2:
        if msg is None:
            raise ValueError("mismatched dimensions: {} != {}", sz1, sz2)
        else:
            raise ValueError("{}: mismatched dimensions: {} != {}", sz1, sz2)
