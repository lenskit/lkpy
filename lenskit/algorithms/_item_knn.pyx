from cpython cimport array
import array
import pandas as pd
import numpy as np
cimport numpy as np
from cython.parallel cimport parallel, prange, threadid
cimport openmp
import logging

_logger = logging.getLogger('_item_knn')

cdef class TmpResults:
    cdef size_t size
    cdef size_t capacity
    cdef array.array items, nbrs, sims

    def __cinit__(self, size_t cap):
        _logger.debug('allocating temporary result holder for %d rows in thread %d',
                      cap, openmp.omp_get_thread_num())
        self.size = 0
        self.capacity = cap
        self.items = array.clone(array.array('q', []), cap, 0)
        self.nbrs = array.clone(array.array('q', []), cap, 0)
        self.sims = array.clone(array.array('d', []), cap, 0)

    cdef void ensure_capacity(self, size_t n) nogil:
        cdef size_t tgt
        if n >= self.capacity:
            tgt = self.capacity * 2
            if n >= tgt:
                tgt = n
            with gil:
                array.resize(self.items, tgt)
                array.resize(self.nbrs, tgt)
                array.resize(self.sims, tgt)


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
    cdef np.int64_t u, i, j, nbr, iidx, uidx
    cdef np.int64_t a, b
    cdef double ur
    cdef double * work_vec
    cdef array.array work_arr
    cdef TmpResults tres
    dbl_tmpl = array.array('d')

    neighborhoods = []

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
        with gil:
            tres = TmpResults(nitems)
            work_arr = array.clone(dbl_tmpl, nitems, 1)
            work_vec = work_arr.data.as_doubles
            _logger.debug('thread %d: work space allocated, ready to go',
                          openmp.omp_get_thread_num())
            
        for i in prange(nitems, schedule='dynamic', chunksize=10):
            for j in range(nitems):
                work_vec[j] = 0

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
                        work_vec[nbr] = work_vec[nbr] + ur * ui_ratings[iidx]

            # now copy the accepted values into the results
            for j in range(nitems):
                if work_vec[j] < threshold: continue
                tres.ensure_capacity(tres.size + 1)
                
                tres.items.data.as_longlongs[tres.size] = i
                tres.nbrs.data.as_longlongs[tres.size] = j
                tres.sims.data.as_doubles[tres.size] = work_vec[j]
                tres.size = tres.size + 1
        
        with gil:
            array.resize(tres.items, tres.size)
            array.resize(tres.nbrs, tres.size)
            array.resize(tres.sims, tres.size)
            _logger.debug('thread %d computed %d pairs', openmp.omp_get_thread_num(), tres.size)
            rframe = pd.DataFrame({'item': tres.items,
                                   'neighbor': tres.nbrs,
                                   'similarity': tres.sims})
            assert len(rframe) == tres.size
            if nnbrs > 0:
                nranks = rframe.groupby('item').similarity.rank(ascending=False)
                rframe = rframe[nranks <= nnbrs]
            neighborhoods.append(rframe)
            _logger.debug('finished parallel item-item build')

    return pd.concat(neighborhoods)
