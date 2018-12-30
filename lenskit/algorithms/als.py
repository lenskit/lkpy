import logging
from collections import namedtuple

import numpy as np
from numba import njit, prange

from . import basic
from .mf_common import BiasMFPredictor, MFPredictor
from ..matrix import sparse_ratings, CSR
from .. import util
from ..math.solve import _dposv

_logger = logging.getLogger(__name__)

Context = namedtuple('Context', [
    'users', 'items',
    'user_matrix', 'item_matrix'
])


@njit(parallel=True, nogil=True)
def _train_matrix(mat: CSR, other: np.ndarray, reg: float):
    "One half of an explicit ALS training round."
    nr = mat.nrows
    nf = other.shape[1]
    regI = np.identity(nf) * reg
    assert mat.ncols == other.shape[0]
    result = np.zeros((nr, nf))
    for i in prange(nr):
        cols = mat.row_cs(i)
        if len(cols) == 0:
            continue

        vals = mat.row_vs(i)
        M = other[cols, :]
        MMT = M.T @ M
        # assert MMT.shape[0] == ctx.n_features
        # assert MMT.shape[1] == ctx.n_features
        A = MMT + regI * len(cols)
        V = M.T @ vals
        # and solve
        _dposv(A, V, True)
        result[i, :] = V

    return result


@njit(parallel=True, nogil=True)
def _train_implicit_matrix(mat: CSR, other: np.ndarray, reg: float):
    "One half of an implicit ALS training round."
    nr = mat.nrows
    nc = other.shape[0]
    nf = other.shape[1]
    assert mat.ncols == nc
    regmat = np.identity(nf) * reg
    Ot = other.T
    OtO = Ot @ other
    OtOr = OtO + regmat
    assert OtO.shape[0] == OtO.shape[1]
    assert OtO.shape[0] == nf
    result = np.zeros((nr, nf))
    for i in prange(nr):
        cols = mat.row_cs(i)
        if len(cols) == 0:
            continue

        rates = mat.row_vs(i)

        # we can optimize by only considering the nonzero entries of Cu-I
        # this means we only need the corresponding matrix columns
        M = other[cols, :]
        # Compute M^T C_u M, restricted to these nonzero entries
        MMT = (M.T.copy() * rates) @ M
        # assert MMT.shape[0] == ctx.n_features
        # assert MMT.shape[1] == ctx.n_features
        # Build the matrix for solving
        A = OtOr + MMT
        # Compute RHS - only used columns (p_ui != 0) values needed
        # Cu is rates + 1 for the cols, so just trim Ot
        y = Ot[:, cols] @ (rates + 1.0)
        # and solve
        _dposv(A, y, True)
        # assert len(uv) == ctx.n_features
        result[i, :] = y

    return result


class BiasedMF(BiasMFPredictor):
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

    Attributes:
        features(int): the number of features.
        iterations(int): the number of training iterations.
        regularization(double): the regularization factor.
        damping(double): the mean damping.
        bias(.basic.Bias): the bias algorithm, or ``True`` to automatically make one.
    """
    timer = None

    def __init__(self, features, *, iterations=20, reg=0.1, damping=5, bias=True):
        self.features = features
        self.iterations = iterations
        self.regularization = reg
        self.damping = damping
        if bias is True:
            self.bias = basic.Bias(damping=damping)
        else:
            self.bias = basic.Bias

    def fit(self, ratings):
        """
        Run ALS to train a model.

        Args:
            ratings: the ratings data frame.

        Returns:
            The algorithm (for chaining).
        """
        self.timer = util.Stopwatch()

        if self.bias is not None:
            _logger.info('[%s] fitting bias model')
            self.bias.fit(ratings)

        current, bias, uctx, ictx = self._initial_model(ratings)

        _logger.info('[%s] training biased MF model with ALS for %d features',
                     self.timer, self.features)
        for epoch, model in enumerate(self._train_iters(current, uctx, ictx)):
            current = model

        _logger.info('trained model in %s', self.timer)

        # unpack and de-Series bias
        gb, ub, ib = bias
        self.global_bias_ = gb
        self.user_bias_ = np.require(ub.values, None, 'C') if ub is not None else None
        self.item_bias_ = np.require(ib.values, None, 'C') if ib is not None else None

        self.item_index_ = current.items
        self.user_index_ = current.users
        self.item_features_ = current.item_matrix
        self.user_features_ = current.user_matrix

        return self

    def _initial_model(self, ratings, bias=None):
        "Initialize a model and build contexts."
        rmat, users, items = sparse_ratings(ratings)
        n_users = len(users)
        n_items = len(items)

        rmat, bias = self._normalize(rmat, users, items)

        _logger.debug('setting up contexts')
        trmat = rmat.transpose()

        _logger.debug('initializing item matrix')
        imat = np.random.randn(n_items, self.features) * 0.01
        umat = np.full((n_users, self.features), np.nan)

        return Context(users, items, umat, imat), bias, rmat, trmat

    def _normalize(self, ratings, users, items):
        "Apply bias normalization to the data in preparation for training."
        n_users = len(users)
        n_items = len(items)
        assert ratings.nrows == n_users
        assert ratings.ncols == n_items

        if self.bias is not None:
            gbias = self.bias.mean_
            ibias = self.bias.item_offsets_
            ubias = self.bias.user_offsets_
        else:
            gbias = 0
            ibias = ubias = None

        _logger.info('[%s] normalizing %dx%d matrix (%d nnz)',
                     self.timer, n_users, n_items, ratings.nnz)
        ratings.values = ratings.values - gbias
        if ibias is not None:
            ibias = ibias.reindex(items, fill_value=0)
            ratings.values = ratings.values - ibias.values[ratings.colinds]
        if ubias is not None:
            ubias = ubias.reindex(users, fill_value=0)
            # create a user index array the size of the data
            reps = np.repeat(np.arange(len(users), dtype=np.int32),
                             ratings.row_nnzs())
            assert len(reps) == ratings.nnz
            # subtract user means
            ratings.values = ratings.values - ubias.values[reps]
            del reps

        return ratings, (gbias, ubias, ibias)

    def _train_iters(self, current, uctx, ictx):
        "Generator of training iterations."
        for epoch in range(self.iterations):
            umat = _train_matrix(uctx, current.item_matrix, self.regularization)
            _logger.debug('[%s] finished user epoch %d', self.timer, epoch)
            imat = _train_matrix(ictx, umat, self.regularization)
            _logger.debug('[%s] finished item epoch %d', self.timer, epoch)
            di = np.linalg.norm(imat - current.item_matrix, 'fro')
            du = np.linalg.norm(umat - current.user_matrix, 'fro')
            _logger.info('[%s] finished epoch %d (|ΔI|=%.3f, |ΔU|=%.3f)', self.timer, epoch, di, du)
            current = current._replace(user_matrix=umat, item_matrix=imat)
            yield current

    def predict_for_user(self, user, items, ratings=None):
        # look up user index
        return self.score_by_ids(user, items)

    def __str__(self):
        return 'als.BiasedMF(features={}, regularization={})'.\
            format(self.features, self.regularization)


class ImplicitMF(MFPredictor):
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

    def fit(self, ratings):
        self.timer = util.Stopwatch()
        current, uctx, ictx = self._initial_model(ratings)

        _logger.info('[%s] training implicit MF model with ALS for %d features',
                     self.timer, self.features)
        _logger.info('have %d observations for %d users and %d items',
                     uctx.nnz, uctx.nrows, ictx.nrows)
        for model in self._train_iters(current, uctx, ictx):
            current = model

        _logger.info('[%s] finished training model with %d features',
                     self.timer, self.features)

        self.item_index_ = current.items
        self.user_index_ = current.users
        self.item_features_ = current.item_matrix
        self.user_features_ = current.user_matrix

        return self

    def _train_iters(self, current, uctx, ictx):
        "Generator of training iterations."
        for epoch in range(self.iterations):
            umat = _train_implicit_matrix(uctx, current.item_matrix,
                                          self.regularization)
            _logger.debug('[%s] finished user epoch %d', self.timer, epoch)
            imat = _train_implicit_matrix(ictx, umat, self.regularization)
            _logger.debug('[%s] finished item epoch %d', self.timer, epoch)
            di = np.linalg.norm(imat - current.item_matrix, 'fro')
            du = np.linalg.norm(umat - current.user_matrix, 'fro')
            _logger.info('[%s] finished epoch %d (|ΔI|=%.3f, |ΔU|=%.3f)', self.timer, epoch, di, du)
            current = current._replace(user_matrix=umat, item_matrix=imat)
            yield current

    def _initial_model(self, ratings):
        "Initialize a model and build contexts."

        rmat, users, items = sparse_ratings(ratings)
        n_users = len(users)
        n_items = len(items)

        _logger.debug('setting up contexts')
        # force values to exist
        if rmat.values is None:
            rmat.values = np.ones(rmat.nnz)
        rmat.values *= self.weight
        trmat = rmat.transpose()

        imat = np.random.randn(n_items, self.features) * 0.01
        imat = np.square(imat)
        umat = np.full((n_users, self.features), np.nan)

        return Context(users, items, umat, imat), rmat, trmat

    def predict_for_user(self, user, items, ratings=None):
        # look up user index
        return self.score_by_ids(user, items)

    def __str__(self):
        return 'als.ImplicitMF(features={}, regularization={}, w={})'.\
            format(self.features, self.regularization, self.weight)
