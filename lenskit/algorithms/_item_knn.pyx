#cython: language_level=3
from cpython cimport array
import array
import pandas as pd
import numpy as np
from scipy import sparse as sps
cimport numpy as np
from numpy cimport math as npm
from cython.parallel cimport parallel, prange, threadid
from libc.stdlib cimport malloc, free, realloc, abort, calloc
from libc.math cimport isnan, fabs
import logging

from lenskit cimport _cy_util as lku

IF OPENMP:
    from openmp cimport omp_get_thread_num, omp_get_num_threads
ELSE:
    cdef int omp_get_thread_num():
        return -1
    cdef int omp_get_num_threads():
        return 0

cdef _logger = logging.getLogger('_item_knn')

cdef class BuildContext:
    cdef readonly int n_users
    cdef readonly int n_items
    cdef readonly matrix
    cdef readonly cscmat

    cdef np.int32_t [:] uptrs
    cdef np.int32_t [:] items
    cdef np.float_t [:] ratings

    cdef np.int32_t [:] r_iptrs
    cdef np.int32_t [:] r_users

    def __cinit__(self, matrix):
        assert np.logical_not(np.isnan(matrix.data)).all()
        self.n_users = matrix.shape[0]
        self.n_items = matrix.shape[1]
        _logger.debug('creating build context for %d users and %d items (%d ratings)',
                      self.n_users, self.n_items, matrix.nnz)

        self.matrix = matrix
        self.cscmat = matrix.tocsc()
        assert sps.isspmatrix_csc(self.cscmat)

        self.uptrs = matrix.indptr
        self.items = matrix.indices
        self.ratings = matrix.data

        self.r_iptrs = self.cscmat.indptr
        self.r_users = self.cscmat.indices


cdef struct ThreadState:
    size_t size
    size_t capacity
    np.int32_t *items
    np.int32_t *nbrs
    np.float_t *sims
    np.float_t *work

cdef ThreadState* tr_new(size_t nitems) nogil:
    cdef ThreadState* tr = <ThreadState*> malloc(sizeof(ThreadState))

    if tr == NULL:
        abort()

    tr.size = 0
    tr.capacity = nitems
    tr.items = <np.int32_t*> malloc(sizeof(np.int32_t) * nitems)
    if tr.items == NULL:
        abort()
    tr.nbrs = <np.int32_t*> malloc(sizeof(np.int32_t) * nitems)
    if tr.nbrs == NULL:
        abort()
    tr.sims = <np.float_t*> malloc(sizeof(np.float_t) * nitems)
    if tr.sims == NULL:
        abort()
    tr.work = <np.float_t*> malloc(sizeof(np.float_t) * nitems)
    if tr.work == NULL:
        abort()

    return tr

cdef void tr_free(ThreadState* self) nogil:
    free(self.items)
    free(self.nbrs)
    free(self.sims)
    free(self.work)
    free(self)

cdef void tr_ensure_capacity(ThreadState* self, size_t n) nogil:
    cdef size_t tgt
    if n > self.capacity:
        tgt = self.capacity * 2
        if n >= tgt:
            tgt = n
        with gil:
            _logger.debug('thread %d: resizing storage from %d to %d',
                          omp_get_thread_num(), self.capacity, tgt)
        self.items = <np.int32_t*> realloc(self.items, sizeof(np.int32_t) * tgt)
        if self.items == NULL:
            abort()
        self.nbrs = <np.int32_t*> realloc(self.nbrs, sizeof(np.int32_t) * tgt)
        if self.nbrs == NULL:
            abort()
        self.sims = <np.float_t*> realloc(self.sims, sizeof(np.float_t) * tgt)
        if self.sims == NULL:
            abort()
        self.capacity = tgt

cdef void tr_add_all(ThreadState* self, int item, size_t nitems,
                     double threshold) nogil:
    cdef int j
    for j in range(nitems):
        if self.work[j] < threshold: continue
        tr_ensure_capacity(self, self.size + 1)
        
        self.items[self.size] = item
        self.nbrs[self.size] = j
        self.sims[self.size] = self.work[j]

        self.size = self.size + 1


cdef void tr_add_nitems(ThreadState* self, int item, size_t nitems,
                        double threshold, int nmax) nogil:
    cdef int* keys
    cdef int j, kn, ki
    cdef np.int64_t nbr
    cdef lku.AccHeap* acc = lku.ah_create(self.work, nmax)
    
    tr_ensure_capacity(self, self.size + nmax)

    kn = 0
    for j in range(nitems):
        if self.work[j] < threshold: continue

        lku.ah_add(acc, j)

    # now that we have the heap built, let us unheap!
    while acc.size > 0:
        nbr = lku.ah_remove(acc)
        self.items[self.size] = item
        self.nbrs[self.size] = nbr
        self.sims[self.size] = self.work[nbr]
        self.size = self.size + 1

    lku.ah_free(acc)


cdef object tr_results(ThreadState* self):
    cdef np.npy_intp size = self.size
    cdef np.ndarray items, nbrs, sims
    items = np.empty(size, dtype=np.int32)
    nbrs = np.empty(size, dtype=np.int32)
    sims = np.empty(size, dtype=np.float_)
    items[:] = <np.int32_t[:self.size]> self.items
    nbrs[:] = <np.int32_t[:self.size]> self.nbrs
    sims[:] = <np.float_t[:self.size]> self.sims
    # items = np.PyArray_SimpleNewFromData(1, &size, np.NPY_INT32, self.items)
    # nbrs = np.PyArray_SimpleNewFromData(1, &size, np.NPY_INT32, self.nbrs)
    # sims = np.PyArray_SimpleNewFromData(1, &size, np.NPY_DOUBLE, self.sims)
    return pd.DataFrame({'item': items, 'neighbor': nbrs, 'similarity': sims})


cpdef sim_matrix(BuildContext context, double threshold, int nnbrs):
    cdef int i
    cdef ThreadState* tres
    cdef list neighborhoods = []

    with nogil, parallel():
        tres = tr_new(context.n_items)
        IF OPENMP:
            with gil:
                _logger.debug('thread %d/%d: starting with context 0x%x',
                            omp_get_thread_num(), omp_get_num_threads(),
                            <unsigned long> tres)
        
        for i in prange(context.n_items, schedule='dynamic', chunksize=10):
            train_row(i, tres, context, threshold, nnbrs)
        
        with gil:
            _logger.debug('thread %d computed %d pairs', omp_get_thread_num(), tres.size)
            if tres.size > 0:
                neighborhoods.append(tr_results(tres))
                _logger.debug('finished parallel item-item build, %d neighbors',
                              len(neighborhoods[-1]))
            else:
                _logger.debug('canceling with no neighbors')
            tr_free(tres)
            tres = NULL
            
    _logger.debug('stacking %d neighborhood frames', len(neighborhoods))
    return pd.concat(neighborhoods, ignore_index=True)


cdef void train_row(int item, ThreadState* tres, BuildContext context,
                    double threshold, int nnbrs) nogil:
    cdef int j, u, uidx, iidx, nbr, urp
    cdef double ur = 0

    lku.zero(tres.work, context.n_items)

    for uidx in range(context.r_iptrs[item], context.r_iptrs[item+1]):
        u = context.r_users[uidx]
        # find user's rating for this item
        urp = -1
        for iidx in range(context.uptrs[u], context.uptrs[u+1]):
            if context.items[iidx] == item:
                urp = iidx
                ur = context.ratings[urp]
                break
        if urp < 0:
            # should never ever get here
            with gil:
                raise AssertionError('failed to find rating (%d,%d)', uidx, item)

        # accumulate pieces of dot products
        for iidx in range(context.uptrs[u], context.uptrs[u+1]):
            nbr = context.items[iidx]
            if nbr != item:
                tres.work[nbr] = tres.work[nbr] + ur * context.ratings[iidx]

    # now copy the accepted values into the results
    if nnbrs > 0:
        tr_add_nitems(tres, item, context.n_items, threshold, nnbrs)
    else:
        tr_add_all(tres, item, context.n_items, threshold)


cpdef predict(matrix, int nitems, int min_nbrs, int max_nbrs,
              np.float_t[:] ratings,
              np.int64_t[:] targets,
              np.float_t[:] scores):
    cdef int[:] indptr = matrix.indptr
    cdef int[:] indices = matrix.indices
    cdef double[:] similarity = matrix.data
    cdef int i, j, iidx, rptr, rend, nidx, nnbrs
    cdef double num, denom

    with nogil:
        for i in range(targets.shape[0]):
            iidx = targets[i]
            rptr = indptr[iidx]
            rend = indptr[iidx + 1]

            num = 0
            denom = 0
            nnbrs = 0
            
            for j in range(rptr, rend):
                nidx = indices[j]
                if isnan(ratings[nidx]):
                    continue
                
                nnbrs = nnbrs + 1
                num = num + ratings[nidx] * similarity[j]
                denom = denom + fabs(similarity[j])

                if max_nbrs > 0 and nnbrs >= max_nbrs:
                    break
                
            if nnbrs < min_nbrs:
                break
            
            scores[iidx] = num / denom

    return None
