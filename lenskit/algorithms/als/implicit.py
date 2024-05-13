from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import torch
from csr import CSR
from numba import njit, prange
from seedbank import numpy_rng

from ... import util
from ...data import sparse_ratings
from ...math.solve import _dposv
from ..bias import Bias
from ..mf_common import MFPredictor
from .common import PartialModel, TrainContext

_log = logging.getLogger(__name__)


class ImplicitMF(MFPredictor):
    """
    Implicit matrix factorization trained with alternating least squares :cite:p:`Hu2008-li`.  This
    algorithm outputs 'predictions', but they are not on a meaningful scale.  If its input
    data contains ``rating`` values, these will be used as the 'confidence' values; otherwise,
    confidence will be 1 for every rated item.

    See the base class :class:`.MFPredictor` for documentation on the estimated parameters
    you can extract from a trained model.

    With weight :math:`w`, this function decomposes the matrix :math:`\\mathbb{1}^* + Rw`, where
    :math:`\\mathbb{1}^*` is an :math:`m \\times n` matrix of all 1s.

    .. versionchanged:: 0.14
        By default, ``ImplicitMF`` ignores a ``rating`` column if one is present in the training
        data.  This can be changed through the ``use_ratings`` option.

    .. versionchanged:: 0.13
        In versions prior to 0.13, ``ImplicitMF`` used the rating column if it was present.
        In 0.13, we added an option to control whether or not the rating column is used; it
        initially defaulted to ``True``, but with a warning.  In 0.14 it defaults to ``False``.

    Args:
        features(int):
            The number of features to train
        iterations(int):
            The number of iterations to train
        reg(float):
            The regularization factor
        weight(float):
            The scaling weight for positive samples (:math:`\\alpha` in :cite:p:`Hu2008-li`).
        use_ratings(bool):
            Whether to use the `rating` column, if present.  Defaults to ``False``; when ``True``,
            the values from the ``rating`` column are used, and multipled by ``weight``; if
            ``False``, ImplicitMF treats every rated user-item pair as having a rating of 1.
        method(str):
            the training method.

            ``'cg'`` (the default)
                Conjugate gradient method :cite:p:`Takacs2011-ix`.
            ``'lu'``
                A direct implementation of the original implicit-feedback ALS concept
                :cite:p:`Hu2008-li` using LU-decomposition to solve for the optimized matrices.

        rng_spec:
            Random number generator or state (see :func:`lenskit.util.random.rng`).
        progress: a :func:`tqdm.tqdm`-compatible progress bar function
    """

    timer = None

    def __init__(
        self,
        features,
        *,
        iterations=20,
        reg=0.1,
        weight=40,
        use_ratings=False,
        method="cg",
        rng_spec=None,
        progress=None,
        save_user_features=True,
    ):
        self.features = features
        self.iterations = iterations
        self.reg = reg
        self.weight = weight
        self.use_ratings = use_ratings
        self.method = method
        self.rng = numpy_rng(rng_spec)
        self.progress = progress if progress is not None else util.no_progress
        self.save_user_features = save_user_features

    def fit(self, ratings, **kwargs):
        util.check_env()
        self.timer = util.Stopwatch()
        for algo in self.fit_iters(ratings, **kwargs):
            pass

        if self.user_features_ is not None:
            _log.info(
                "[%s] finished training model with %d features (|P|=%f, |Q|=%f)",
                self.timer,
                self.features,
                np.linalg.norm(self.user_features_, "fro"),
                np.linalg.norm(self.item_features_, "fro"),
            )
        else:
            _log.info(
                "[%s] finished training model with %d features (|Q|=%f)",
                self.timer,
                self.features,
                np.linalg.norm(self.item_features_, "fro"),
            )

        # unpack the regularization
        if isinstance(self.reg, tuple):
            ureg, ireg = self.reg
        else:
            ureg = self.reg

        # compute OtOr and save it on the model
        self.OtOr_ = _implicit_otor(self.item_features_, ureg)

        return self

    def fit_iters(self, ratings, **kwargs):
        current, uctx, ictx = self._initial_model(ratings)

        _log.info(
            "[%s] training implicit MF model with ALS for %d features", self.timer, self.features
        )
        _log.info(
            "have %d observations for %d users and %d items", uctx.nnz, uctx.nrows, ictx.nrows
        )
        for model in self._train_iters(current, uctx, ictx):
            self._save_model(model)
            yield self

    def _save_model(self, model):
        self.item_index_ = model.items
        self.user_index_ = model.users
        self.item_features_ = model.item_matrix
        if self.save_user_features:
            self.user_features_ = model.user_matrix
        else:
            self.user_features_ = None

    def _train_iters(self, current, uctx, ictx):
        "Generator of training iterations."
        if self.method == "lu":
            train = _train_implicit_lu
        elif self.method == "cg":
            train = _train_implicit_cg
        else:
            raise ValueError("unknown solver " + self.method)

        if isinstance(self.reg, tuple):
            ureg, ireg = self.reg
        else:
            ureg = ireg = self.reg

        for epoch in self.progress(range(self.iterations), desc="ImplicitMF", leave=False):
            du = train(uctx, current.user_matrix, current.item_matrix, ureg)
            _log.debug("[%s] finished user epoch %d", self.timer, epoch)
            di = train(ictx, current.item_matrix, current.user_matrix, ireg)
            _log.debug("[%s] finished item epoch %d", self.timer, epoch)
            _log.info("[%s] finished epoch %d (|ΔP|=%.3f, |ΔQ|=%.3f)", self.timer, epoch, du, di)
            yield current

    def _initial_model(self, ratings):
        "Initialize a model and build contexts."

        if not self.use_ratings:
            ratings = ratings[["user", "item"]]

        rmat, users, items = sparse_ratings(ratings)
        n_users = len(users)
        n_items = len(items)

        _log.debug("setting up contexts")
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
        if ratings is not None and len(ratings) > 0:
            ri_idxes = self.item_index_.get_indexer_for(ratings.index)
            ri_good = ri_idxes >= 0
            ri_it = ri_idxes[ri_good]
            if self.use_ratings is False:
                ri_val = np.ones(len(ri_good))
            else:
                ri_val = ratings.values[ri_good]
            ri_val *= self.weight
            u_feat = _train_implicit_row_lu(ri_it, ri_val, self.item_features_, self.OtOr_)
            return self.score_by_ids(user, items, u_feat)
        else:
            # look up user index
            return self.score_by_ids(user, items)

    def __str__(self):
        return "als.ImplicitMF(features={}, reg={}, w={})".format(
            self.features, self.reg, self.weight
        )


@njit
def _inplace_axpy(a, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]


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


@njit(nogil=True)
def _implicit_otor(other, reg):
    nf = other.shape[1]
    regmat = np.identity(nf)
    regmat *= reg
    Ot = other.T
    OtO = Ot @ other
    OtO += regmat
    return OtO


@njit(parallel=True, nogil=True)
def _train_implicit_cg(mat, this: np.ndarray, other: np.ndarray, reg: float):
    "One half of an implicit ALS training round with conjugate gradient."
    nr = mat.nrows
    nc = other.shape[0]

    assert mat.ncols == nc

    OtOr = _implicit_otor(other, reg)

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
def _train_implicit_lu(mat: CSR, this: np.ndarray, other: np.ndarray, reg: float):
    "One half of an implicit ALS training round."
    nr = mat.nrows
    nc = other.shape[0]
    assert mat.ncols == nc
    OtOr = _implicit_otor(other, reg)
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
        y = other.T[:, cols] @ (rates + 1.0)
        # and solve
        _dposv(A, y, True)
        # assert len(uv) == ctx.n_features
        delta = this[i, :] - y
        frob += np.dot(delta, delta)
        this[i, :] = y

    return np.sqrt(frob)


@njit(nogil=True)
def _train_implicit_row_lu(items, ratings, other, otOr):
    """
    Args:
        items(np.ndarray[i64]): the item IDs the user has rated
        ratings(np.ndarray): the user's (normalized) ratings for those items
        other(np.ndarray): the item-feature matrix
        reg(float): the regularization term
    Returns:
        np.ndarray: the user-feature vector (equivalent to V in the current LU code)
    """
    # we can optimize by only considering the nonzero entries of Cu-I
    # this means we only need the corresponding matrix columns
    M = other[items, :]
    # Compute M^T (C_u-I) M, restricted to these nonzero entries
    MMT = (M.T.copy() * ratings) @ M
    # Build the matrix for solving
    A = otOr + MMT
    # Compute RHS - only used columns (p_ui != 0) values needed
    # Cu is rates + 1 for the cols, so just trim Ot
    y = other.T[:, items] @ (ratings + 1.0)
    # and solve
    _dposv(A, y, True)

    return y
