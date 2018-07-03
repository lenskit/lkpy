import numpy as np
cimport numpy as np

cimport scipy.linalg.cython_blas as blas

import logging

_logger = logging.getLogger('lenskit.algorithms.funksvd')

cpdef void train_unclamped(np.int64_t[:] users, np.int64_t[:] items, np.float_t[:] ratings, np.float_t[:] bias,
                      np.float_t[:,::1] umat, np.float_t[:,::1] imat,
                      int niters, double lrate, double reg) nogil:
    cdef int nfeatures = umat.shape[1]
    cdef size_t nsamples = users.shape[0]
    cdef double pred, error, ufv, ifv, ufd, ifd, sse
    cdef np.int64_t user, item
    cdef int f, epoch, s
    cdef int inc = 1

    with gil:
        assert items.shape[0] == nsamples
        assert ratings.shape[0] == nsamples
        assert bias.shape[0] == nsamples
        assert umat.shape[1] == nfeatures

    for f in range(nfeatures):
        for epoch in range(niters):
            sse = 0
            for s in range(nsamples):
                user = users[s]
                item = items[s]
                pred = bias[s] + blas.ddot(&nfeatures, &umat[user,0], &inc, &imat[item,0], &inc)
                error = ratings[s] - pred
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
