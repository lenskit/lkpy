"""
Array utilities.
"""

from numba import njit


@njit
def swap(a, i, j):
    t = a[i]
    a[i] = a[j]
    a[j] = t
