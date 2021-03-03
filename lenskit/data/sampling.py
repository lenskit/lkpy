import numpy as np
from numba import njit


@njit
def sample_unweighted(mat):
    return np.random.randint(0, mat.ncols)


@njit
def sample_weighted(mat):
    j = np.random.randint(0, mat.nnz)
    return mat.colinds[j]


@njit(nogil=True)
def neg_sample(mat, uv, sample):
    """
    Sample the examples from a user-item matrix.  For each user in uv, it samples an
    item that they have not rated using rejection sampling.

    While this is embarassingly parallel, we do not parallelize because it's usually
    used in parallel.
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
