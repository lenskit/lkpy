import logging
from collections import namedtuple

import pandas as pd
import numpy as np
from numba import njit

from . import basic
from . import Predictor, Trainable
from .mf_common import BiasMFModel
from ..matrix import sparse_ratings

_logger = logging.getLogger(__package__)

_Ctx = namedtuple('_Ctx', [
    'n_users', 'n_items', 'n_features',
    'uptrs', 'items', 'ui_ratings',
    'iptrs', 'users', 'iu_ratings'
])


def _train_users(ctx: _Ctx, imat: np.ndarray, reg: float):
    umat = np.zeros((ctx.n_users, ctx.n_features))
    for u in range(ctx.n_users):
        sp = ctx.uptrs[u]
        ep = ctx.uptrs[u+1]
        if sp == ep:
            continue

        items = ctx.items[sp:ep]
        M = imat[items, :]
        MMT = M.T @ M
        assert MMT.shape[0] == ctx.n_features
        assert MMT.shape[1] == ctx.n_features
        A = MMT + np.identity(ctx.n_features) * reg * (ep - sp)
        Ainv = np.linalg.inv(A)
        V = M.T @ ctx.ui_ratings[sp:ep]
        uv = Ainv @ V
        assert len(uv) == ctx.n_features
        umat[u, :] = uv

    return umat


def _train_items(ctx: _Ctx, umat: np.ndarray, reg: float):
    imat = np.zeros((ctx.n_items, ctx.n_features))
    for i in range(ctx.n_items):
        sp = ctx.iptrs[i]
        ep = ctx.iptrs[i+1]
        if sp == ep:
            continue

        users = ctx.users[sp:ep]
        M = umat[users, :]
        MMT = M.T @ M
        assert MMT.shape[0] == ctx.n_features
        assert MMT.shape[1] == ctx.n_features
        A = MMT + np.identity(ctx.n_features) * reg * (ep - sp)
        Ainv = np.linalg.inv(A)
        V = M.T @ ctx.iu_ratings[sp:ep]
        iv = Ainv @ V
        assert len(iv) == ctx.n_features
        imat[i, :] = iv

    return imat


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

    def __init__(self, features, iterations=10, reg=0.015, damping=5):
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

        if bias is None:
            _logger.info('training bias model')
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
        _logger.info('normalizing %dx%d matrix (%d nnz)', n_users, n_items, rmat.nnz)
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
        ctx = _Ctx(n_users, n_items, self.features,
                   rmat.indptr, rmat.indices, rmat.data,
                   trmat.indptr, trmat.indices, trmat.data)

        _logger.debug('initializing item matrix')
        imat = np.random.randn(n_items, self.features) * 0.1
        umat = np.zeros((n_users, self.features))

        _logger.info('training biased MF model with ALS for %d features', self.features)
        for epoch in range(self.iterations):
            umat2 = _train_users(ctx, imat, self.regularization)
            _logger.info('finished user epoch %d (|Δ|=%.5f)',
                         epoch, np.linalg.norm(umat2 - umat, 'fro'))
            umat = umat2
            imat2 = _train_items(ctx, umat, self.regularization)
            _logger.info('finished item epoch %d (|Δ|=%.5f)',
                         epoch, np.linalg.norm(imat2 - imat, 'fro'))
            imat = imat2

        _logger.info('finished model training')

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
