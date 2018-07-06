from cpython cimport array
import array
import pandas as pd
import numpy as np
cimport numpy as np
from cython.parallel cimport parallel, prange, threadid
from libc.stdlib cimport malloc, free, realloc, abort
cimport openmp
import logging

_logger = logging.getLogger('_item_knn')

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
        self.n_users = matrix.shape[0]
        self.n_items = matrix.shape[1]
        _logger.debug('creating build context for %d users and %d items (%d ratings)',
                      self.n_users, self.n_items, matrix.nnz)

        self.matrix = matrix
        self.cscmat = matrix.tocsc(copy=False)

        self.uptrs = matrix.indptr
        self.items = matrix.indices
        self.ratings = matrix.data

        self.r_iptrs = self.cscmat.indptr
        self.r_users = self.cscmat.indices


cdef struct ThreadState:
    size_t size
    size_t capacity
    np.int64_t *items
    np.int64_t *nbrs
    np.float_t *sims
    np.float_t *work

cdef ThreadState* tr_new(size_t nitems) nogil:
    cdef ThreadState* tr = <ThreadState*> malloc(sizeof(ThreadState))

    if tr == NULL:
        abort()

    tr.size = 0
    tr.capacity = nitems
    tr.items = <np.int64_t*> malloc(sizeof(np.int64_t) * nitems)
    if tr.items == NULL:
        abort()
    tr.nbrs = <np.int64_t*> malloc(sizeof(np.int64_t) * nitems)
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
        self.items = <np.int64_t*> realloc(self.items, sizeof(np.int64_t) * tgt)
        if self.items == NULL:
            abort()
        self.nbrs = <np.int64_t*> realloc(self.nbrs, sizeof(np.int64_t) * tgt)
        if self.nbrs == NULL:
            abort()
        self.sims = <np.float_t*> realloc(self.sims, sizeof(np.float_t) * tgt)
        if self.sims == NULL:
            abort()
        self.capacity = tgt


cpdef sim_matrix(BuildContext context, double threshold, int nnbrs):
    cdef int i
    cdef ThreadState* tres

    neighborhoods = []

    with nogil, parallel():
        tres = tr_new(context.n_items)
        
        for i in prange(context.n_items, schedule='dynamic', chunksize=10):
            train_row(i, tres, context, threshold, nnbrs)
        
        with gil:
            _logger.debug('thread %d computed %d pairs', openmp.omp_get_thread_num(), tres.size)
            if tres.size > 0:
                rframe = pd.DataFrame({'item': np.asarray(<np.int64_t[:tres.size]> tres.items).copy(),
                                       'neighbor': np.asarray(<np.int64_t[:tres.size]> tres.nbrs).copy(),
                                       'similarity': np.asarray(<np.float_t[:tres.size]> tres.sims).copy()})
                assert len(rframe) == tres.size
                if nnbrs > 0:
                    _logger.debug('thread %d: trimming neighborhoods',
                                  openmp.omp_get_thread_num())
                    nranks = rframe.groupby('item').similarity.rank(ascending=False, method='first')
                    rframe = rframe[nranks <= nnbrs]
                neighborhoods.append(rframe)
            tr_free(tres)
            _logger.debug('finished parallel item-item build')

    _logger.debug('stacking %d neighborhood frames', len(neighborhoods))
    return pd.concat(neighborhoods, ignore_index=True)

cdef void train_row(int item, ThreadState* tres, BuildContext context,
                    double threshold, int nnbrs) nogil:
    cdef np.int64_t j, u, uidx, iidx, nbr
    cdef double ur = 0

    for j in range(context.n_items):
        tres.work[j] = 0

    for uidx in range(context.r_iptrs[item], context.r_iptrs[item+1]):
        u = context.r_users[uidx]
        # find user's rating for this item
        for iidx in range(context.uptrs[u], context.uptrs[u+1]):
            if context.items[iidx] == item:
                ur = context.ratings[iidx]
                break

        # accumulate pieces of dot products
        for iidx in range(context.uptrs[u], context.uptrs[u+1]):
            nbr = context.items[iidx]
            if nbr != item:
                tres.work[nbr] = tres.work[nbr] + ur * context.ratings[iidx]

    # now copy the accepted values into the results
    for j in range(context.n_items):
        if tres.work[j] < threshold: continue
        tr_ensure_capacity(tres, tres.size + 1)
        
        tres.items[tres.size] = item
        tres.nbrs[tres.size] = j
        tres.sims[tres.size] = tres.work[j]
        tres.size = tres.size + 1
