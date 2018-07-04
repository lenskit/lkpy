import pandas as pd
import numpy as np
cimport numpy as np
from cython.parallel cimport parallel, prange
from libc.stdlib cimport abort, malloc, free
import logging

_logger = logging.getLogger('_item_knn')


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


cpdef sim_matrix(int nusers, int nitems,
                 np.int64_t[:] iu_items, np.int64_t[:] iu_users,
                 np.int64_t[:] ui_users, np.int64_t[:] ui_items, np.float_t[:] ui_ratings,
                 double threshold, int nnbrs):
    iu_istart_v = np.zeros(nitems + 1, dtype=np.int64)
    cdef np.int64_t[:] iu_istart = iu_istart_v
    ui_ustart_v = np.zeros(nusers + 1, dtype=np.int64)
    cdef np.int64_t[:] ui_ustart = ui_ustart_v
    cdef np.float_t * value 
    cdef np.int64_t u, i, j, nbr, iidx, uidx
    cdef np.int64_t a, b
    cdef double ur
    neighborhoods = {}

    assert iu_istart.shape[0] == nitems + 1
    assert ui_ustart.shape[0] == nusers + 1
    assert iu_items.shape[0] == iu_users.shape[0]

    # set up the item & user start records
    for a in range(iu_items.shape[0]):
        b = iu_items[a]
        if iu_istart[b] == 0 and b > 0:
            # update
            iu_istart[b] = a
    iu_istart[nitems] = iu_items.shape[0]
    for a in range(ui_users.shape[0]):
        b = ui_users[a]
        if ui_ustart[b] == 0 and b > 0:
            # update
            ui_ustart[b] = a
    ui_ustart[nusers] = ui_users.shape[0]

    with nogil, parallel():
        values = <np.float_t*> malloc(sizeof(np.float_t) * nitems)
        if values == NULL:
            abort()
            
        for i in prange(nitems, schedule='dynamic', chunksize=10):
            for j in range(nitems):
                values[j] = 0

            for uidx in range(iu_istart[i], iu_istart[i+1]):
                u = iu_users[uidx]
                # find user's rating for this item
                for iidx in range(ui_ustart[u], ui_ustart[u+1]):
                    if ui_items[iidx] == i:
                        ur = ui_ratings[iidx]
                        break
                # accumulate pieces of dot products
                for iidx in range(ui_ustart[u], ui_ustart[u+1]):
                    nbr = ui_items[iidx]
                    if nbr != i:
                        values[nbr] = values[nbr] + ur * ui_ratings[iidx]
            with gil:
                series = pd.Series(np.asarray(<np.float_t[:nitems]> values))
                series = series[series >= threshold]
                if nnbrs > 0:
                    series = series.nlargest(nnbrs)
                _logger.debug('found %d neighbors for item %d', len(series), i)
                neighborhoods[i] = series
        
        with gil:
            _logger.debug('finished parallel item-item build')
        free(values)

    _logger.debug('built neighborhoods for %d items', len(neighborhoods))
    return neighborhoods
