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

    def __cinit__(self, np.ndarray umat, np.ndarray imat):
        self.user_features = umat
        self.item_features = imat
        self.umat = self.user_features
        self.imat = self.item_features

    @staticmethod
    def fresh(int feature_count, int nusers, int nitems, double init=0.1):
        umat = np.full([nusers, feature_count], init, dtype=np.float_)
        imat = np.full([nitems, feature_count], 0.1, dtype=np.float_)
        return Model(umat, imat)


cdef class Kernel:
    cdef double score(self, Model model, int user, int item, double base) nogil:
        pass

cdef class DotKernel(Kernel):
    cdef double score(self, Model model, int user, int item, double base) nogil:
        cdef int nfeatures = model.feature_count
        cdef int inc = 1

        return base + blas.ddot(&nfeatures, &model.umat[user,0], &inc, &model.imat[item,0], &inc)

cdef class ClampKernel(Kernel):
    cdef double min
    cdef double max

    def __cinit__(self, min, max):
        self.min = min
        self.max = max

    cdef double score(self, Model model, int user, int item, double base) nogil:
        cdef double res = base
        for i in range(model.feature_count):
            res += model.umat[user,i] * model.imat[item,i]
            if res < self.min:
                res = self.min
            elif res > self.max:
                res = self.max

        return res

cpdef double score(Kernel kern, Model model, int user, int item, double base):
    return kern.score(model, user, item, base)

cpdef void train(Context ctx, Params params, Model model, Kernel kernel) nogil:
    cdef double pred, error, ufv, ifv, ufd, ifd, sse
    cdef np.int64_t user, item
    cdef int f, epoch, s
    cdef int inc = 1

    for f in range(model.feature_count):
        for epoch in range(params.iter_count):
            sse = 0
            for s in range(ctx.n_samples):
                user = ctx.u_v[s]
                item = ctx.i_v[s]
                pred = kernel.score(model, user, item, ctx.b_v[s])
                error = ctx.r_v[s] - pred
                sse += error * error
                
                # compute deltas
                ufv = model.umat[user, f]
                ifv = model.umat[item, f]
                ufd = error * ifv - params.reg_term * ufv
                ifd = error * ufv - params.reg_term * ifv
                model.umat[user, f] += ufd * params.learning_rate
                model.imat[item, f] += ifd * params.learning_rate

            with gil:
                _logger.debug('finished epoch %d for feature %d (RMSE=%f)',
                              epoch, f, np.sqrt(sse))
