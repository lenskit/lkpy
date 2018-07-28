#cython: language_level=3
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
        assert model.feature_count == feature_count
        assert model.user_count == nusers
        assert model.item_count == nitems
        return model


cpdef double score(Model model, int user, int item, double base) nogil:
    cdef int nfeatures = model.feature_count
    cdef int inc = 1

    return base + blas.ddot(&nfeatures, &model.umat[user,0], &inc, &model.imat[item,0], &inc)


cpdef void train(Context ctx, Params params, Model model, domain):
    cdef double pred, error, ufv, ifv, ufd, ifd, sse
    cdef np.int64_t user, item
    cdef int f, epoch, s
    cdef double reg = params.reg_term
    cdef double lrate = params.learning_rate
    cdef np.int64_t[:] u_v = ctx.users
    cdef np.int64_t[:] i_v = ctx.items
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
            for epoch in range(params.iter_count):
                sse = 0
                for s in range(ctx.n_samples):
                    user = u_v[s]
                    item = i_v[s]
                    pred = score(model, user, item, b_v[s])
                    if pred < rmin:
                        pred = rmin
                    elif pred > rmax:
                        pred = rmax

                    error = r_v[s] - pred
                    sse += error * error
                    
                    # compute deltas
                    ufv = umat[user, f]
                    ifv = imat[item, f]
                    ufd = error * ifv - reg * ufv
                    ifd = error * ufv - reg * ifv
                    umat[user, f] += ufd * lrate
                    imat[item, f] += ifd * lrate

        _logger.debug('finished feature %d (RMSE=%f)', f, np.sqrt(sse / ctx.n_samples))
