import logging
from collections import namedtuple

import numpy as np
from numba import njit, prange

from . import basic
from .mf_common import BiasMFPredictor, MFPredictor
from ..matrix import sparse_ratings
from .. import util
from ..math.solve import _dposv

_logger = logging.getLogger(__name__)

PartialModel = namedtuple('PartialModel', [
    'users', 'items',
    'user_matrix', 'item_matrix'
])


@njit
def _inplace_axpy(a, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]


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

    nd = len(w)
    Xt = X.T[:, xis]
    resid = w @ Xt
    resid *= -1.0
    resid += y

    for e in range(epochs):
        for k in range(nd):
            xk = Xt[k, :]
            num = np.dot(xk, resid) - reg * w[k]
            denom = np.dot(xk, xk) + reg
            dw = num / denom
            w[k] += dw
            _inplace_axpy(-dw, xk, resid)


@njit(parallel=True, nogil=True)
def _train_matrix_cd(mat, this: np.ndarray, other: np.ndarray, reg: float):
    """
    One half of an explicit ALS training round using coordinate descent.

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

        w = this[i, :].copy()
        _rr_solve(other, cols, vals, w, reg * len(cols), 2)
        delta = this[i, :] - w
        frob += np.dot(delta, delta)
        this[i, :] = w

    return np.sqrt(frob)


@njit(parallel=True, nogil=True)
def _train_matrix_lu(mat, this: np.ndarray, other: np.ndarray, reg: float):
    """
    One half of an explicit ALS training round using LU-decomposition on the normal
    matrices to solve the least squares problem.

    Args:
        mat: the :math:`m \\times n` matrix of ratings
        this: the :math:`m \\times k` matrix to train
        other: the :math:`n \\times k` matrix of sample features
        reg: the regularization term
    """
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


@njit
def _cg_a_mult(OtOr, X, y, v):
    """
    Compute the multiplication Av, where A = X'X + X'yX + λ.
    """
    XtXv = OtOr @ v
    XtyXv = X.T @ (y * (X @ v))
    return XtXv + XtyXv


@njit
def _cg_solve(OtOr, X, y, w, epochs):
    """
    Use conjugate gradient method to solve the system M†(X'X + X'yX + λ)w = M†X'(y+1).
    The parameter OtOr = X'X + λ.
    """
    nf = X.shape[1]
    # compute inverse of the Jacobi preconditioner
    Ad = np.diag(OtOr).copy()
    for i in range(X.shape[0]):
        for k in range(nf):
            Ad[k] += X[i, k] * y[i] * X[i, k]

    iM = np.reciprocal(Ad)

    # compute residuals
    b = X.T @ (y + 1.0)
    r = _cg_a_mult(OtOr, X, y, w)
    r *= -1
    r += b

    # compute initial values
    z = iM * r
    p = z

    # and solve
    for i in range(epochs):
        gam = np.dot(r, z)
        Ap = _cg_a_mult(OtOr, X, y, p)
        al = gam / np.dot(p, Ap)
        _inplace_axpy(al, p, w)
        _inplace_axpy(-al, Ap, r)
        z = iM * r
        bet = np.dot(r, z) / gam
        p = z + bet * p


@njit(parallel=True, nogil=True)
def _train_implicit_cg(mat, this: np.ndarray, other: np.ndarray, reg: float):
    "One half of an implicit ALS training round with conjugate gradient."
    nr = mat.nrows
    nc = other.shape[0]
    nf = other.shape[1]

    assert mat.ncols == nc

    regmat = np.identity(nf) * reg
    Ot = other.T
    OtO = Ot @ other
    OtOr = OtO + regmat

    frob = 0.0

    for i in prange(nr):
        cols = mat.row_cs(i)
        if len(cols) == 0:
            continue

        rates = mat.row_vs(i)

        # we can optimize by only considering the nonzero entries of Cu-I
        # this means we only need the corresponding matrix columns
        M = other[cols, :]
        # and solve
        w = this[i, :].copy()
        _cg_solve(OtOr, M, rates, w, 3)

        # update stats
        delta = this[i, :] - w
        frob += np.dot(delta, delta)

        # put back the result
        this[i, :] = w

    return np.sqrt(frob)


@njit(parallel=True, nogil=True)
def _train_implicit_lu(mat, this: np.ndarray, other: np.ndarray, reg: float):
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
    frob = 0.0

    for i in prange(nr):
        cols = mat.row_cs(i)
        if len(cols) == 0:
            continue

        rates = mat.row_vs(i)

        # we can optimize by only considering the nonzero entries of Cu-I
        # this means we only need the corresponding matrix columns
        M = other[cols, :]
        # Compute M^T (C_u-I) M, restricted to these nonzero entries
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
        delta = this[i, :] - y
        frob += np.dot(delta, delta)
        this[i, :] = y

    return np.sqrt(frob)


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
        rng_spec:
            Random number generator or state (see :func:`lenskit.util.random.rng`).
        progress: a :func:`tqdm.tqdm`-compatible progress bar function
    """
    timer = None

    def __init__(self, features, *, iterations=20, reg=0.1, damping=5, bias=True, method='cd',
                 rng_spec=None, progress=None):
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
        self.rng = util.rng(rng_spec)

    def fit(self, ratings, **kwargs):
        """
        Run ALS to train a model.

        Args:
            ratings: the ratings data frame.

        Returns:
            The algorithm (for chaining).
        """

        self.timer = util.Stopwatch()

        for epoch, algo in enumerate(self.fit_iters(ratings, **kwargs)):
            pass  # we just need to do the iterations

        _logger.info('trained model in %s (|P|=%f, |Q|=%f)', self.timer,
                     np.linalg.norm(self.user_features_, 'fro'),
                     np.linalg.norm(self.item_features_, 'fro'))

        del self.timer
        return self

    def fit_iters(self, ratings, **kwargs):
        """
        Run ALS to train a model, returning each iteration as a generator.

        Args:
            ratings: the ratings data frame.

        Returns:
            The algorithm (for chaining).
        """

        if self.bias:
            _logger.info('[%s] fitting bias model', self.timer)
            self.bias.fit(ratings)

        current, bias, uctx, ictx = self._initial_model(ratings)

        # unpack and de-Series bias
        gb, ub, ib = bias
        self.global_bias_ = gb
        self.user_bias_ = np.require(ub.values, None, 'C') if ub is not None else None
        self.item_bias_ = np.require(ib.values, None, 'C') if ib is not None else None

        _logger.info('[%s] training biased MF model with ALS for %d features',
                     self.timer, self.features)
        for epoch, model in enumerate(self._train_iters(current, uctx, ictx)):
            self._save_params(model)
            yield self

    def _save_params(self, model):
        "Save the parameters into model attributes."
        self.item_index_ = model.items
        self.user_index_ = model.users
        self.item_features_ = model.item_matrix
        self.user_features_ = model.user_matrix

    def _initial_model(self, ratings, bias=None):
        "Initialize a model and build contexts."
        rmat, users, items = sparse_ratings(ratings)
        n_users = len(users)
        n_items = len(items)

        rmat, bias = self._normalize(rmat, users, items)

        _logger.debug('setting up contexts')
        trmat = rmat.transpose()

        _logger.debug('initializing item matrix')
        imat = self.rng.standard_normal((n_items, self.features))
        imat /= np.linalg.norm(imat, axis=1).reshape((n_items, 1))
        _logger.debug('|Q|: %f', np.linalg.norm(imat, 'fro'))
        _logger.debug('initializing user matrix')
        umat = self.rng.standard_normal((n_users, self.features))
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

    def __getstate__(self):
        state = dict(self.__dict__)
        if state.get('bias', None):  # don't save bias training
            state['bias'] = util.clone(self.bias)
        return state

    def __str__(self):
        return 'als.BiasedMF(features={}, regularization={})'.\
            format(self.features, self.regularization)


class ImplicitMF(MFPredictor):
    """
    Implicit matrix factorization trained with alternating least squares [HKV2008]_.  This
    algorithm outputs 'predictions', but they are not on a meaningful scale.  If its input
    data contains ``rating`` values, these will be used as the 'confidence' values; otherwise,
    confidence will be 1 for every rated item.

    ``'cd'`` (the default)
        Conjugate gradient method [TPT2011]_.
    ``'lu'``
        A direct implementation of the original implicit-feedback ALS concept [HKV2008]_ using
        LU-decomposition to solve for the optimized matrices.

    See the base class :class:`.MFPredictor` for documentation on the estimated parameters
    you can extract from a trained model.

    .. [HKV2008] Y. Hu, Y. Koren, and C. Volinsky. 2008.
       Collaborative Filtering for Implicit Feedback Datasets.
       In _Proceedings of the 2008 Eighth IEEE International Conference on Data Mining_, 263–272.
       DOI `10.1109/ICDM.2008.22 <http://dx.doi.org/10.1109/ICDM.2008.22>`_

    .. [TPT2011] Gábor Takács, István Pilászy, and Domonkos Tikk. 2011. Applications of the
        Conjugate Gradient Method for Implicit Feedback Collaborative Filtering.

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train
        reg(double): the regularization factor
        weight(double): the scaling weight for positive samples (:math:`\\alpha` in [HKV2008]_).
        rng_spec:
            Random number generator or state (see :func:`lenskit.util.random.rng`).
        progress: a :func:`tqdm.tqdm`-compatible progress bar function
    """
    timer = None

    def __init__(self, features, *, iterations=20, reg=0.1, weight=40, method='cg',
                 rng_spec=None, progress=None):
        self.features = features
        self.iterations = iterations
        self.reg = reg
        self.weight = weight
        self.method = method
        self.rng = util.rng(rng_spec)
        self.progress = progress if progress is not None else util.no_progress

    def fit(self, ratings, **kwargs):
        self.timer = util.Stopwatch()
        for algo in self.fit_iters(ratings, **kwargs):
            pass

        _logger.info('[%s] finished training model with %d features (|P|=%f, |Q|=%f)',
                     self.timer, self.features,
                     np.linalg.norm(self.user_features_, 'fro'),
                     np.linalg.norm(self.item_features_, 'fro'))

        return self

    def fit_iters(self, ratings, **kwargs):
        current, uctx, ictx = self._initial_model(ratings)

        _logger.info('[%s] training implicit MF model with ALS for %d features',
                     self.timer, self.features)
        _logger.info('have %d observations for %d users and %d items',
                     uctx.nnz, uctx.nrows, ictx.nrows)
        for model in self._train_iters(current, uctx, ictx):
            self._save_model(model)
            yield self

    def _save_model(self, model):
        self.item_index_ = model.items
        self.user_index_ = model.users
        self.item_features_ = model.item_matrix
        self.user_features_ = model.user_matrix

    def _train_iters(self, current, uctx, ictx):
        "Generator of training iterations."
        if self.method == 'lu':
            train = _train_implicit_lu
        elif self.method == 'cg':
            train = _train_implicit_cg
        else:
            raise ValueError('unknown solver ' + self.method)

        if isinstance(self.reg, tuple):
            ureg, ireg = self.reg
        else:
            ureg = ireg = self.reg

        for epoch in self.progress(range(self.iterations), desc='ImplicitMF', leave=False):
            du = train(uctx.N, current.user_matrix, current.item_matrix, ureg)
            _logger.debug('[%s] finished user epoch %d', self.timer, epoch)
            di = train(ictx.N, current.item_matrix, current.user_matrix, ireg)
            _logger.debug('[%s] finished item epoch %d', self.timer, epoch)
            _logger.info('[%s] finished epoch %d (|ΔP|=%.3f, |ΔQ|=%.3f)', self.timer, epoch, du, di)
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

        imat = self.rng.standard_normal((n_items, self.features)) * 0.01
        imat = np.square(imat)
        umat = self.rng.standard_normal((n_users, self.features)) * 0.01
        umat = np.square(umat)

        return PartialModel(users, items, umat, imat), rmat, trmat

    def predict_for_user(self, user, items, ratings=None):
        # look up user index
        return self.score_by_ids(user, items)

    def __str__(self):
        return 'als.ImplicitMF(features={}, reg={}, w={})'.\
            format(self.features, self.reg, self.weight)
