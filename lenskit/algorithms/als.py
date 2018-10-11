import logging
from collections import namedtuple

import pandas as pd
import numpy as np
from numba import njit, jitclass, prange, float64, int32, int64

from . import basic
from . import Predictor, Trainable
from .mf_common import BiasMFModel
from ..matrix import sparse_ratings
from .. import util

_logger = logging.getLogger(__name__)


@jitclass({
    'n_rows': int64,
    'n_features': int64,
    'ptrs': int32[:],
    'cols': int32[:],
    'vals': float64[:]
})
class _Ctx:
    """
    An input matrix (CSR) for training users or items.

    Attributes:
        n_rows(int): the number of rows (users or items).
        n_features(int): the number of features to train.
        ptrs(array): the row pointers for the CSR matrix.
        cols(array): the column indices for the CSR matrix.
        vals(array): the data values for the CSR matrix.
    """
    def __init__(self, nr, nf, ps, cs, vs):
        self.n_rows = nr
        self.n_features = nf
        self.ptrs = ps
        self.cols = cs
        self.vals = vs


@njit(parallel=True, nogil=True)
def _train_matrix(ctx: _Ctx, other: np.ndarray, reg: float):
    result = np.zeros((ctx.n_rows, ctx.n_features))
    for i in prange(ctx.n_rows):
        sp = ctx.ptrs[i]
        ep = ctx.ptrs[i+1]
        if sp == ep:
            continue

        cols = ctx.cols[sp:ep]
        M = other[cols, :]
        MMT = M.T @ M
        # assert MMT.shape[0] == ctx.n_features
        # assert MMT.shape[1] == ctx.n_features
        A = MMT + np.identity(ctx.n_features) * reg * (ep - sp)
        Ainv = np.linalg.inv(A)
        V = M.T @ ctx.vals[sp:ep]
        uv = Ainv @ V
        # assert len(uv) == ctx.n_features
        result[i, :] = uv

    return result


class BiasedMF(Predictor, Trainable):
    """
    Algorithm class implementing biased matrix factorization with regularized alternating
    least squares.

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train
        reg(double): the regularization factor
        damping(double): damping factor for the underlying mean
    """

    def __init__(self, features, iterations=10, reg=0.1, damping=5):
        self.features = features
        self.iterations = iterations
        self.regularization = reg
        self.damping = damping

    def train(self, ratings, bias=None):
        """
        Run ALS to train a model.

        Args:
            ratings: the ratings data frame.
            bias(.bias.BiasModel): a pre-trained bias model to use.

        Returns:
            BiasMFModel: The trained biased MF model.
        """
        timer = util.Stopwatch()

        if bias is None:
            _logger.info('[%s] training bias model', timer)
            bias = basic.Bias(damping=self.damping).train(ratings)
        # unpack the bias
        if isinstance(bias, basic.BiasModel):
            gbias = bias.mean
            ibias = bias.items
            ubias = bias.users
        else:
            # we have a single global bias (for e.g. implicit feedback data)
            gbias = bias
            ibias = None
            ubias = None

        rmat, users, items = sparse_ratings(ratings)
        n_users = len(users)
        n_items = len(items)
        _logger.info('[%s] normalizing %dx%d matrix (%d nnz)',
                     timer, n_users, n_items, rmat.nnz)
        rmat.data = rmat.data - gbias
        if ibias is not None:
            ibias = ibias.reindex(items)
            rmat.data = rmat.data - ibias.values[rmat.indices]
        if ubias is not None:
            ubias = ubias.reindex(users)
            # create a user index array the size of the data
            reps = np.repeat(np.arange(len(users), dtype=np.int32),
                             np.diff(rmat.indptr))
            assert len(reps) == rmat.nnz
            # subtract user means
            rmat.data = rmat.data - ubias.values[reps]
            del reps

        _logger.debug('setting up context')
        trmat = rmat.tocsc()
        uctx = _Ctx(n_users, self.features,
                    rmat.indptr, rmat.indices, rmat.data)
        ictx = _Ctx(n_items, self.features,
                    trmat.indptr, trmat.indices, trmat.data)

        _logger.debug('initializing item matrix')
        imat = np.random.randn(n_items, self.features) * 0.1
        umat = np.zeros((n_users, self.features))

        _logger.info('[%s] training biased MF model with ALS for %d features',
                     timer, self.features)
        for epoch in range(self.iterations):
            umat2 = _train_matrix(uctx, imat, self.regularization)
            _logger.info('[%s] finished user epoch %d (|Δ|=%.5f)',
                         timer, epoch, np.linalg.norm(umat2 - umat, 'fro'))
            umat = umat2
            imat2 = _train_matrix(ictx, umat, self.regularization)
            _logger.info('[%s] finished item epoch %d (|Δ|=%.5f)',
                         timer, epoch, np.linalg.norm(imat2 - imat, 'fro'))
            imat = imat2

        _logger.info('trained model in %s', timer)

        return BiasMFModel(users, items, gbias, ubias, ibias, umat, imat)

    def predict(self, model, user, items, ratings=None):
        # look up user index
        uidx = model.lookup_user(user)
        if uidx < 0:
            _logger.debug('user %s not in model', user)
            return pd.Series(np.nan, index=items)

        # get item index & limit to valid ones
        items = np.array(items)
        iidx = model.lookup_items(items)
        good = iidx >= 0
        good_items = items[good]
        good_iidx = iidx[good]

        # multiply
        _logger.debug('scoring %d items for user %s', len(good_items), user)
        rv = model.score(uidx, good_iidx)

        res = pd.Series(rv, index=good_items)
        res = res.reindex(items)
        return res

    def __str__(self):
        return 'als.BiasedMF(features={}, regularization={})'.format(self.features, self.regularization)
