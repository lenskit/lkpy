#cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np
from numpy cimport math as npm

cimport scipy.linalg.cython_blas as blas

import logging

_logger = logging.getLogger('lenskit.algorithms.funksvd')


cdef class Context:
    cdef readonly np.ndarray users
    cdef readonly np.ndarray items
    cdef readonly np.ndarray ratings
    cdef readonly np.ndarray bias
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


cdef class Params:
    cdef readonly int iter_count
    cdef readonly double learning_rate
    cdef readonly double reg_term

    def __cinit__(self, niters, lrate, reg):
        self.iter_count = niters
        self.learning_rate = lrate
        self.reg_term = reg


cdef class Model:
    cdef readonly np.ndarray user_features
    cdef np.float_t[:,::1] umat
    cdef readonly np.ndarray item_features
    cdef np.float_t[:,::1] imat
    cdef readonly int feature_count
    cdef readonly int user_count
    cdef readonly int item_count
    cdef readonly double initial_value

    def __cinit__(self, np.ndarray umat, np.ndarray imat):
        self.user_features = umat
        self.item_features = imat
        self.umat = self.user_features
        self.imat = self.item_features
        self.feature_count = umat.shape[1]
        self.user_count = umat.shape[0]
        self.item_count = imat.shape[0]
        assert imat.shape[1] == self.feature_count

    @staticmethod
    def fresh(int feature_count, int nusers, int nitems, double init=0.1):
        umat = np.full([nusers, feature_count], init, dtype=np.float_)
        imat = np.full([nitems, feature_count], init, dtype=np.float_)
        model = Model(umat, imat)
        model.initial_value = init
        assert model.feature_count == feature_count
        assert model.user_count == nusers
        assert model.item_count == nitems
        return model


@cython.boundscheck(False)
cpdef double score(Model model, int user, int item, double base) nogil:
    cdef int nfeatures = model.feature_count
    cdef int inc = 1
    cdef int f
    cdef double sum = base
    for f in range(model.feature_count):
        sum += model.umat[user,f] * model.imat[item,f]
    return sum
    # return base + blas.ddot(&nfeatures, &model.umat[user,0], &inc, &model.imat[item,0], &inc)


# @cython.boundscheck(False)
cpdef void train(Context ctx, Params params, Model model, domain):
    cdef double pred, error, ufv, ifv, ufd, ifd, sse, acc_ud, acc_id
    cdef np.int64_t user, item
    cdef int f, epoch, s
    cdef double reg = params.reg_term
    cdef double lrate = params.learning_rate
    cdef double trail
    cdef np.ndarray est = ctx.bias.copy()
    cdef np.float_t[:] est_v = est
    cdef int[:] u_v = ctx.users
    cdef int[:] i_v = ctx.items
    cdef np.float_t[:] r_v = ctx.ratings
    cdef np.float_t[:] b_v = ctx.bias
    cdef np.float_t[:,::1] umat = model.user_features
    cdef np.float_t[:,::1] imat = model.item_features
    cdef double rmin, rmax
    if domain is None:
        rmin = -npm.INFINITY
        rmax = npm.INFINITY
    else:
        rmin, rmax = domain

    _logger.debug('feature count: %d', model.feature_count)
    _logger.debug('iteration count: %d', params.iter_count)
    _logger.debug('learning rate: %f', params.learning_rate)
    _logger.debug('regularization: %f', params.reg_term)
    _logger.debug('samples: %d', ctx.n_samples)

    for f in range(model.feature_count):
        with nogil:
            trail = model.initial_value * model.initial_value * (model.feature_count - f - 1)
            for epoch in range(params.iter_count):
                sse = 0
                acc_ud = 0
                acc_id = 0
                for s in range(ctx.n_samples):
                    user = u_v[s]
                    item = i_v[s]
                    ufv = umat[user, f]
                    ifv = imat[item, f]

                    pred = est_v[s] + ufv * ifv + trail
                    if pred < rmin:
                        pred = rmin
                    elif pred > rmax:
                        pred = rmax

                    error = r_v[s] - pred
                    sse += error * error
                    
                    # compute deltas
                    ufd = error * ifv - reg * ufv
                    ufd = ufd * lrate
                    acc_ud += ufd * ufd
                    ifd = error * ufv - reg * ifv
                    ifd = ifd * lrate
                    acc_id += ifd * ifd
                    umat[user, f] += ufd
                    imat[item, f] += ifd
                
                with gil:
                    _logger.debug('finished epoch %d:%d (RMSE=%f, |Δu|=%f, |Δi|=%f)',
                                  f, epoch, np.sqrt(sse / ctx.n_samples),
                                  np.sqrt(acc_ud), np.sqrt(acc_id))
        
        _logger.info('finished feature %d (RMSE=%f)', f, np.sqrt(sse / ctx.n_samples))
        est = est + model.user_features[ctx.users, f] * model.item_features[ctx.items, f]
        est = np.maximum(est, rmin)
        est = np.minimum(est, rmax)
        est_v = est

        
