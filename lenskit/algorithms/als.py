import logging
from collections import namedtuple

import pandas as pd
import numpy as np
from numba import njit, jitclass, prange, float64, int32, int64

from . import basic
from . import Predictor, Trainable
from .mf_common import BiasMFModel, MFModel
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
    Biased matrix factorization trained with alternating least squares [ZWSP2008]_.  This is a
    prediction-oriented algorithm suitable for explicit feedback data.

    .. [ZWSP2008] Yunhong Zhou, Dennis Wilkinson, Robert Schreiber, and Rong Pan. 2008.
        Large-Scale Parallel Collaborative Filtering for the Netflix Prize.
        In +Algorithmic Aspects in Information and Management_, LNCS 5034, 337–348.
        DOI `10.1007/978-3-540-68880-8_32 <http://dx.doi.org/10.1007/978-3-540-68880-8_32>`_.

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train
        reg(double): the regularization factor
        damping(double): damping factor for the underlying mean
    """
    timer = None

    def __init__(self, features, iterations=20, reg=0.1, damping=5):
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
        self.timer = util.Stopwatch()

        current, uctx, ictx = self._initial_model(ratings, bias)

        _logger.info('[%s] training biased MF model with ALS for %d features',
                     self.timer, self.features)
        for epoch, model in enumerate(self._train_iters(current, uctx, ictx)):
            current = model

        _logger.info('trained model in %s', self.timer)

        return current

    def _initial_model(self, ratings, bias=None):
        "Initialize a model and build contexts."
        gbias, ubias, ibias = self._get_bias(bias, ratings)
        rmat, users, items = sparse_ratings(ratings)
        n_users = len(users)
        n_items = len(items)

        rmat, ubias, ibias = self._normalize(rmat, users, items, gbias, ubias, ibias)
        assert len(ubias) == n_users and len(ibias) == n_items

        _logger.debug('setting up contexts')
        uctx = _Ctx(n_users, self.features,
                    rmat.indptr, rmat.indices, rmat.data)
        trmat = rmat.tocsc()
        ictx = _Ctx(n_items, self.features,
                    trmat.indptr, trmat.indices, trmat.data)

        _logger.debug('initializing item matrix')
        imat = np.random.randn(n_items, self.features) * 0.01

        return BiasMFModel(users, items, gbias, ubias, ibias, None, imat), uctx, ictx

    def _get_bias(self, bias, ratings):
        "Extract or construct bias terms for the model."
        if bias is None:
            _logger.info('[%s] training bias model', self.timer)
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

        return gbias, ubias, ibias

    def _normalize(self, ratings, users, items, gbias, ubias, ibias):
        "Apply bias normalization to the data in preparation for training."
        n_users = len(users)
        n_items = len(items)
        _logger.info('shape %s, u %d, i %d', ratings.shape, n_users, n_items)
        assert ratings.shape[0] == n_users
        assert ratings.shape[1] == n_items

        _logger.info('[%s] normalizing %dx%d matrix (%d nnz)',
                     self.timer, n_users, n_items, ratings.nnz)
        ratings.data = ratings.data - gbias
        if ibias is not None:
            ibias = ibias.reindex(items)
            ratings.data = ratings.data - ibias.values[ratings.indices]
        if ubias is not None:
            ubias = ubias.reindex(users)
            # create a user index array the size of the data
            reps = np.repeat(np.arange(len(users), dtype=np.int32),
                             np.diff(ratings.indptr))
            assert len(reps) == ratings.nnz
            # subtract user means
            ratings.data = ratings.data - ubias.values[reps]
            del reps

        return ratings, ubias, ibias

    def _train_iters(self, current, uctx, ictx):
        "Generator of training iterations."
        for epoch in range(self.iterations):
            umat = _train_matrix(uctx, current.item_features, self.regularization)
            _logger.debug('[%s] finished user epoch %d', self.timer, epoch)
            imat = _train_matrix(ictx, umat, self.regularization)
            _logger.debug('[%s] finished item epoch %d', self.timer, epoch)
            di = np.linalg.norm(imat - current.item_features, 'fro')
            if current.user_features is not None:
                du = np.linalg.norm(umat - current.user_features, 'fro')
            else:
                du = np.nan
            _logger.info('[%s] finished epoch %d (|ΔI|=%.3f, |ΔU|=%.3f)', self.timer, epoch, di, du)
            current = BiasMFModel(current.user_index, current.item_index,
                                  current.global_bias, current.user_bias, current.item_bias,
                                  umat, imat)
            yield current

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
        return 'als.BiasedMF(features={}, regularization={})'.\
            format(self.features, self.regularization)


class ImplicitMF(Predictor, Trainable):
    """
    Implicit matrix factorization trained with alternating least squares [HKV2008]_.  This
    algorithm outputs 'predictions', but they are not on a meaningful scale.  If its input
    data contains ``rating`` values, these will be used as the 'confidence' values; otherwise,
    confidence will be 1 for every rated item.

    .. [HKV2008] Y. Hu, Y. Koren, and C. Volinsky. 2008.
       Collaborative Filtering for Implicit Feedback Datasets.
       In _Proceedings of the 2008 Eighth IEEE International Conference on Data Mining_, 263–272.
       DOI `10.1109/ICDM.2008.22 <http://dx.doi.org/10.1109/ICDM.2008.22>`_

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train
        reg(double): the regularization factor
        weight(double): the scaling weight for positive samples (:math:`\\alpha` in [HKV2008]_).
    """
    timer = None

    def __init__(self, features, iterations=20, reg=0.1, weight=40):
        self.features = features
        self.iterations = iterations
        self.regularization = reg
        self.weight = weight

    def train(self, ratings):
        self.timer = util.Stopwatch()
        initial, uctx, ictx = self._initial_model(ratings)

    def _initial_model(self, ratings):
        "Initialize a model and build contexts."

        rmat, users, items = sparse_ratings(ratings)
        n_users = len(users)
        n_items = len(items)

        _logger.debug('setting up contexts')
        uctx = _Ctx(n_users, self.features,
                    rmat.indptr, rmat.indices, rmat.data)
        trmat = rmat.tocsc()
        ictx = _Ctx(n_items, self.features,
                    trmat.indptr, trmat.indices, trmat.data)

        imat = np.random.randn(n_items, self.features) * 0.01
        imat = np.square(imat)

        return MFModel(users, items, None, imat), uctx, ictx
