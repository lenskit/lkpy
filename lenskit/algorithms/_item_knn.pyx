import pandas as pd
import numpy as np
cimport numpy as np
from cython.parallel cimport parallel, prange
from libc.stdlib cimport abort, malloc, free

cpdef double sparse_dot(int [:] ks1, double [:] vs1, int[:] ks2, double [:] vs2) nogil:
    cdef double sum = 0
    cdef size_t n1 = ks1.shape[0]
    cdef size_t n2 = ks2.shape[0]
    cdef int i1 = 0
    cdef int i2 = 0
    cdef int k1, k2

    while i1 < n1 and i2 < n2:
        k1 = ks1[i1]
        k2 = ks2[i2]
        if k1 < k2:
            i1 += 1
        elif k2 < k1:
            i2 += 1
        else:
            sum += vs1[i1] * vs2[i2]
            i1 += 1
            i2 += 1

    return sum


cpdef sim_matrix(int[:] iu_items, int[:] iu_users,
                 int[:] ui_users, int[:] ui_items, double[:] ui_ratings,
                 double threshold, int nnbrs):
    # iu_items has the starting position of each item's users, plus the length
    cdef int nitems = iu_items.shape[0] - 1
    # ui_users has the starting position of each user's items, plus the length
    cdef int nusers = ui_users.shape[0] - 1
    cdef np.float_t * values
    cdef int u, i, nbr, iidx, uidx
    cdef double ur
    neighborhoods = {}

    with nogil, parallel():
        values = <np.float_t*> malloc(sizeof(np.float_t) * nitems)
        if values == NULL:
            abort()
            
        for i in prange(nitems, schedule='dynamic', chunksize=10):
            for uidx in range(iu_items[i], iu_items[i+1]):
                u = iu_users[uidx]
                # find user's rating for this item
                for iidx in range(ui_users[u], ui_users[i+1]):
                    if ui_items[iidx] == i:
                        ur = ui_ratings[iidx]
                        break
                # accumulate pieces of dot products
                for iidx in range(ui_users[u], ui_users[i+1]):
                    nbr = ui_items[iidx]
                    if nbr != i:
                        values[nbr] = values[nbr] + ur * ui_ratings[iidx]
            with gil:
                series = pd.Series(np.asarray(<np.float_t[:nitems]> values))
                series = series[series >= threshold]
                if nnbrs > 0:
                    series = series.nlargest(nnbrs)
                neighborhoods[i] = series
        
        free(values)

    return neighborhoods
