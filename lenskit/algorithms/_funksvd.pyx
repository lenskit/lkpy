import numpy as np
cimport numpy as np

cimport scipy.linalg.cython_blas as blas

import logging

_logger = logging.getLogger('lenskit.algorithms.funksvd')


cdef class Context:
    cdef readonly np.ndarray users
    cdef np.int64_t[:] u_v
    cdef readonly np.ndarray items
    cdef np.int64_t[:] i_v
    cdef readonly np.ndarray ratings
    cdef np.float_t[:] r_v
    cdef readonly np.ndarray bias
    cdef np.float_t[:] b_v
    cdef readonly size_t n_samples

    def __cinit__(self, users, items, ratings, bias):
        self.users = users
        self.items = items
        self.ratings = ratings
        self.bias = bias
        self.n_samples = users.shape[0]

        assert items.shape[0] == self.n_samples
        assert ratings.shape[0] == self.n_samples
        assert bias.shape[0] == self.n_samples

        self.u_v = self.users
        self.i_v = self.items
        self.r_v = self.ratings
        self.b_v = self.bias


cpdef void train_unclamped(Context ctx, np.float_t[:,::1] umat, np.float_t[:,::1] imat,
                           int niters, double lrate, double reg) nogil:
    cdef int nfeatures = umat.shape[1]
    cdef double pred, error, ufv, ifv, ufd, ifd, sse
    cdef np.int64_t user, item
    cdef int f, epoch, s
    cdef int inc = 1

    with gil:
        assert umat.shape[1] == nfeatures
        assert imat.shape[1] == nfeatures

    for f in range(nfeatures):
        for epoch in range(niters):
            sse = 0
            for s in range(ctx.n_samples):
                user = ctx.u_v[s]
                item = ctx.i_v[s]
                pred = ctx.b_v[s] + blas.ddot(&nfeatures, &umat[user,0], &inc, &imat[item,0], &inc)
                error = ctx.r_v[s] - pred
                sse += error * error
                
                # compute deltas
                ufv = umat[user, f]
                ifv = umat[item, f]
                ufd = error * ifv - reg * ufv
                ifd = error * ufv - reg * ifv
                umat[user, f] += ufd * lrate
                imat[item, f] += ifd * lrate

            with gil:
                _logger.debug('finished epoch %d for feature %d (RMSE=%f)',
                              epoch, f, np.sqrt(sse))
