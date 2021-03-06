import numpy as np
from numba import njit


@njit
def sample_unweighted(mat):
    """
    Candidate sampling function for use with :py:func:`neg_sample`.
    It samples items uniformly at random.
    """
    return np.random.randint(0, mat.ncols)


@njit
def sample_weighted(mat):
    """
    Candidate sampling function for use with :py:func:`neg_sample`.
    It samples items proportionally to their popularity.
    """
    j = np.random.randint(0, mat.nnz)
    return mat.colinds[j]


@njit(nogil=True)
def neg_sample(mat, uv, sample):
    """
    Sample the examples from a user-item matrix.  For each user in ``uv``, it samples
    an item that they have not rated using rejection sampling.

    While this is embarassingly parallel, we do not parallelize because it's often
    used in parallel.

    This returns both the items and the sample counts for debugging::

        neg_items, counts = neg_sample(matrix, users, sample_unweighted)

    Args:
        mat(csr.CSR):
            The user-item matrix.  Its values are ignored and do not need to be present.
        uv(numpy.ndarray):
            An array of user IDs.
        sample(function):
            A sampling function to sample candidate negative items.  Should be one of
            :py:func:`sample_weighted` or :py:func:`sample_unweighted`.

    Returns:
        numpy.ndarray, numpy.ndarray:
            Two arrays:

            1.  The sampled negative item IDs.
            2.  An array of sample counts, the number of samples required to sample each
                item.  This is useful for diagnosing sample inefficiency.
    """
    n = len(uv)
    jv = np.empty(n, dtype=np.int32)
    sc = np.ones(n, dtype=np.int32)

    for i in range(n):
        u = uv[i]
        used = mat.row_cs(u)
        j = sample(mat)
        while np.any(used == j):
            j = sample(mat)
            sc[i] = sc[i] + 1
        jv[i] = j

    return jv, sc
