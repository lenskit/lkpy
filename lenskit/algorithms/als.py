import logging
from collections import namedtuple

import numpy as np
from numba import njit, prange

from . import basic
from .mf_common import BiasMFPredictor, MFPredictor
from ..matrix import sparse_ratings, _CSR
from .. import util
from ..math.solve import _dposv

_logger = logging.getLogger(__name__)

PartialModel = namedtuple('PartialModel', [
    'users', 'items',
    'user_matrix', 'item_matrix'
])


@njit
def _rr_solve(X, xis, y, w, reg, epochs):
    """
    RR1 coordinate descent solver.

    Args:
        X(ndarray): The feature matrix.
        xis(ndarray): Row numbers in ``X`` that are rated.
        y(ndarray): Rating values corresponding to ``xis``.
        w(ndarray): Input/output vector to solve.
    """

    nr = len(xis)
    nd = len(w)
    resid = y.copy()

    for i in range(nr):
        resid[i] -= np.dot(X[xis[i], :], w)

    for e in range(epochs):
        for k in range(nd):
            xk = X[xis, k]
            num = np.dot(xk, resid) - reg * w[k]
            denom = np.dot(xk, xk) + reg
            dw = num / denom
            w[k] += dw
            resid -= dw * xk


@njit(parallel=True, nogil=True)
def _train_matrix_cd(mat: _CSR, this: np.ndarray, other: np.ndarray, reg: float):
    """
    One half of an explicit ALS training round.

    Args:
        mat: the :math:`m \\times n` matrix of ratings
        this: the :math:`m \\times k` matrix to train
        other: the :math:`n \\times k` matrix of sample features
        reg: the regularization term
    """
    nr = mat.nrows
    nf = other.shape[1]
    assert mat.ncols == other.shape[0]
    assert mat.nrows == this.shape[0]
    assert this.shape[1] == nf

    frob = 0.0

    for i in prange(nr):
        cols = mat.row_cs(i)
        if len(cols) == 0:
            continue

        vals = mat.row_vs(i)

        w = this[i, :]
        delta = w.copy()
        _rr_solve(other, cols, vals, w, reg * len(cols), 2)
        delta -= w
        frob += np.dot(delta, delta)
        this[i, :] = w

    return np.sqrt(frob)


@njit(parallel=True, nogil=True)
def _train_matrix_lu(mat: _CSR, this: np.ndarray, other: np.ndarray, reg: float):
    "One half of an explicit ALS training round."
    nr = mat.nrows
    nf = other.shape[1]
    regI = np.identity(nf) * reg
    assert mat.ncols == other.shape[0]
    frob = 0.0

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
        delta = this[i, :] - V
        frob += np.dot(delta, delta)
        this[i, :] = V

    return np.sqrt(frob)


@njit(parallel=True, nogil=True)
def _train_implicit_matrix(mat: _CSR, other: np.ndarray, reg: float):
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

    It provides two solvers for the optimization step (the `method` parameter):

    ``'cd'`` (the default)
        Coordinate descent [TPT2011]_, adapted for a separately-trained bias model and to use
        weighted regularization as in the original ALS paper [ZWSP2008]_.
    ``'lu'``
        A direct implementation of the original ALS concept [ZWSP2008]_ using LU-decomposition
        to solve for the optimized matrices.

    See the base class :class:`.BiasMFPredictor` for documentation on
    the estimated parameters you can extract from a trained model.

    .. [ZWSP2008] Yunhong Zhou, Dennis Wilkinson, Robert Schreiber, and Rong Pan. 2008.
        Large-Scale Parallel Collaborative Filtering for the Netflix Prize.
        In +Algorithmic Aspects in Information and Management_, LNCS 5034, 337–348.
        DOI `10.1007/978-3-540-68880-8_32 <http://dx.doi.org/10.1007/978-3-540-68880-8_32>`_.

    .. [TPT2011] Gábor Takács, István Pilászy, and Domonkos Tikk. 2011. Applications of the
        Conjugate Gradient Method for Implicit Feedback Collaborative Filtering.

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train
        reg(float): the regularization factor; can also be a tuple ``(ureg, ireg)`` to
            specify separate user and item regularization terms.
        damping(float): damping factor for the underlying mean
        bias(bool or :class:`Bias`): the bias model.  If ``True``, fits a :class:`Bias` with
            damping ``damping``.
        method(str): the solver to use (see above).
        progress: a :func:`tqdm.tqdm`-compatible progress bar function
    """
    timer = None

    def __init__(self, features, *, iterations=20, reg=0.1, damping=5, bias=True, method='cd',
                 progress=None):
        self.features = features
        self.iterations = iterations
        self.regularization = reg
        self.damping = damping
        self.method = method
        if bias is True:
            self.bias = basic.Bias(damping=damping)
        else:
            self.bias = bias
        self.progress = progress if progress is not None else util.no_progress

    def fit(self, ratings):
        """
        Run ALS to train a model.

        Args:
            ratings: the ratings data frame.

        Returns:
            The algorithm (for chaining).
        """
        self.timer = util.Stopwatch()

        if self.bias:
            _logger.info('[%s] fitting bias model', self.timer)
            self.bias.fit(ratings)

        current, bias, uctx, ictx = self._initial_model(ratings)

        _logger.info('[%s] training biased MF model with ALS for %d features',
                     self.timer, self.features)
        for epoch, model in enumerate(self._train_iters(current, uctx, ictx)):
            current = model

        _logger.info('trained model in %s (|P|=%f, |Q|=%f)', self.timer,
                     np.linalg.norm(current.user_matrix, 'fro'),
                     np.linalg.norm(current.item_matrix, 'fro'))

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
        imat = np.random.randn(n_items, self.features)
        imat /= np.linalg.norm(imat, axis=1).reshape((n_items, 1))
        _logger.debug('|Q|: %f', np.linalg.norm(imat, 'fro'))
        _logger.debug('initializing user matrix')
        umat = np.random.randn(n_users, self.features)
        umat /= np.linalg.norm(umat, axis=1).reshape((n_users, 1))
        _logger.debug('|P|: %f', np.linalg.norm(umat, 'fro'))

        return PartialModel(users, items, umat, imat), bias, rmat, trmat

    def _normalize(self, ratings, users, items):
        "Apply bias normalization to the data in preparation for training."
        n_users = len(users)
        n_items = len(items)
        assert ratings.nrows == n_users
        assert ratings.ncols == n_items

        if self.bias:
            gbias = self.bias.mean_
            ibias = self.bias.item_offsets_
            ubias = self.bias.user_offsets_
            if ratings.values is None:
                ratings.N.values = np.ones(np.nnz)
        else:
            gbias = 0
            ibias = ubias = None
            return ratings, (gbias, ibias, ubias)

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
        """
        Generator of training iterations.

        Args:
            current(PartialModel): the current model step.
            uctx(ndarray): the user-item rating matrix for training user features.
            ictx(ndarray): the item-user rating matrix for training item features.
        """
        n_items = len(current.items)
        n_users = len(current.users)
        assert uctx.nrows == n_users
        assert uctx.ncols == n_items
        assert ictx.nrows == n_items
        assert ictx.ncols == n_users

        if self.method == 'cd':
            train = _train_matrix_cd
        elif self.method == 'lu':
            train = _train_matrix_lu
        else:
            raise ValueError('invalid training method ' + self.method)

        if isinstance(self.regularization, tuple):
            ureg, ireg = self.regularization
        else:
            ureg = ireg = self.regularization

        for epoch in self.progress(range(self.iterations), desc='BiasedMF', leave=False):
            du = train(uctx.N, current.user_matrix, current.item_matrix, ureg)
            _logger.debug('[%s] finished user epoch %d', self.timer, epoch)
            di = train(ictx.N, current.item_matrix, current.user_matrix, ireg)
            _logger.debug('[%s] finished item epoch %d', self.timer, epoch)
            _logger.info('[%s] finished epoch %d (|ΔP|=%.3f, |ΔQ|=%.3f)', self.timer, epoch, du, di)
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

    See the base class :class:`.MFPredictor` for documentation on the estimated parameters
    you can extract from a trained model.

    .. [HKV2008] Y. Hu, Y. Koren, and C. Volinsky. 2008.
       Collaborative Filtering for Implicit Feedback Datasets.
       In _Proceedings of the 2008 Eighth IEEE International Conference on Data Mining_, 263–272.
       DOI `10.1109/ICDM.2008.22 <http://dx.doi.org/10.1109/ICDM.2008.22>`_

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train
        reg(double): the regularization factor
        weight(double): the scaling weight for positive samples (:math:`\\alpha` in [HKV2008]_).
        progress: a :func:`tqdm.tqdm`-compatible progress bar function
    """
    timer = None

    def __init__(self, features, *, iterations=20, reg=0.1, weight=40, progress=None):
        self.features = features
        self.iterations = iterations
        self.reg = reg
        self.weight = weight
        self.progress = progress if progress is not None else util.no_progress

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
        for epoch in self.progress(range(self.iterations), desc='ImplicitMF', leave=False):
            umat = _train_implicit_matrix(uctx.N, current.item_matrix,
                                          self.reg)
            _logger.debug('[%s] finished user epoch %d', self.timer, epoch)
            imat = _train_implicit_matrix(ictx.N, umat, self.reg)
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

        return PartialModel(users, items, umat, imat), rmat, trmat

    def predict_for_user(self, user, items, ratings=None):
        # look up user index
        return self.score_by_ids(user, items)

    def __str__(self):
        return 'als.ImplicitMF(features={}, reg={}, w={})'.\
            format(self.features, self.reg, self.weight)
