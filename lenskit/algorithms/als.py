# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import math
import warnings
from collections import namedtuple
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as tmp
from csr import CSR
from numba import njit, prange
from seedbank import numpy_rng

from .. import util
from ..data import sparse_ratings
from ..math.solve import _dposv
from .bias import Bias
from .mf_common import MFPredictor

_logger = logging.getLogger(__name__)
__trainers: tuple[ALSCholeskyTrainer, ALSCholeskyTrainer]
TrainHalf: TypeAlias = Literal["left", "right"]

PartialModel = namedtuple("PartialModel", ["users", "items", "user_matrix", "item_matrix"])


@njit
def _inplace_axpy(a, x, y):
    for i in range(len(x)):
        y[i] += a * x[i]


def _train_solve_row(
    cols: torch.Tensor,
    vals: torch.Tensor,
    this: torch.Tensor,
    other: torch.Tensor,
    regI: torch.Tensor,
) -> torch.Tensor:
    nf = this.shape[1]
    M = other[cols, :]
    MMT = M.T @ M
    # assert MMT.shape[0] == ctx.n_features
    # assert MMT.shape[1] == ctx.n_features
    A = MMT + regI * len(cols)
    V = M.T @ vals
    V = V.reshape(1, nf, 1)
    # and solve
    L, info = torch.linalg.cholesky_ex(A)
    if int(info):
        raise RuntimeError("error computing Cholesky decomposition (not symmetric?)")
    V = torch.cholesky_solve(V, L).reshape(nf)
    return V


def _worker_init(left, right):
    global __trainers
    __trainers = left, right
    torch.set_num_threads(1)


def _worker_row(input):
    # convert to numpy so we don't waste shm resources on results
    left, right = __trainers
    mode, row = input
    match mode:
        case "left":
            trainer = left
        case "right":
            trainer = right

    return row, trainer.train_row(row)


class ALSCholeskyTrainer:
    matrix: torch.Tensor
    nrows: int
    ncols: int
    left: torch.Tensor
    right: torch.Tensor
    reg: float
    embed_size: int
    device: torch.device
    dtype: torch.dtype

    regI: torch.Tensor | None = None

    def __init__(
        self, mat: torch.Tensor, left: torch.Tensor, right: torch.Tensor, reg: float
    ) -> None:
        self.matrix = mat
        self.nrows, self.ncols = mat.shape
        self.left = left
        self.right = right
        self.reg = reg
        self.embed_size = right.shape[1]
        self.device = left.device
        self.dtype = left.dtype
        assert right.dtype == left.dtype
        assert right.device == left.device

    def train_row(self, i: int) -> float:
        row = self.matrix[i]
        (n,) = row.shape
        if n == 0:
            return 0.0

        cols = row.indices()[0].detach()
        vals = row.values().detach()

        if self.regI is None:
            self.regI = torch.eye(self.embed_size, device=self.device, dtype=self.dtype)
            self.regI *= self.reg

        V = _train_solve_row(cols, vals, self.left, self.right, self.regI)
        delta = self.left[i, :] - V
        self.left[i, :] = V
        return float(torch.dot(delta, delta))

    def update_row(self, i, V) -> float:
        delta = self.left[i, :] - V
        self.left[i, :] = V
        return torch.dot(delta, delta).item()


def _train_sequential(trainer: ALSCholeskyTrainer) -> float:
    frob = 0.0

    for i in range(trainer.nrows):
        V = trainer.train_row(i)
        frob += trainer.update_row(i, V)

    return math.sqrt(frob)


def _train_rows(mat: torch.Tensor, mode: TrainHalf):
    nr, nc = mat.shape
    for i in range(nr):
        row = mat[i]
        if row.shape[0] > 0:
            cols = row.indices()[0].numpy()
            vals = row.values().numpy()
            yield mode, i, cols, vals


def _train_parallel(pool, trainer: ALSCholeskyTrainer, mode: TrainHalf) -> float:
    frob = 0.0

    for i, diff in pool.imap(_worker_row, ((mode, x) for x in range(trainer.nrows))):
        frob += diff

    return math.sqrt(frob)


def _train_matrix_cholesky(mat: torch.Tensor, this: torch.Tensor, other: torch.Tensor, reg: float):
    """
    One half of an explicit ALS training round using Cholesky decomposition on the normal
    matrices to solve the least squares problem.

    Args:
        mat: the :math:`m \\times n` matrix of ratings
        this: the :math:`m \\times k` matrix to train
        other: the :math:`n \\times k` matrix of sample features
        reg: the regularization term
    """
    nr, nc = mat.shape
    nf = other.shape[1]
    regI = torch.eye(nf, device=this.device, dtype=this.dtype) * reg
    frob = 0.0

    for i in range(nr):
        row = mat[i]
        (n,) = row.shape
        if n == 0:
            continue

        cols = row.indices()[0]
        vals = row.values().type(this.type())

        V = _train_solve_row(cols, vals, this, other, regI)
        delta = this[i, :] - V
        frob += torch.dot(delta, delta)
        this[i, :] = V

    return np.sqrt(frob)


def _train_bias_row_cholesky(
    items: torch.Tensor, ratings: torch.Tensor, other: torch.Tensor, reg: float
) -> torch.Tensor:
    """
    Args:
        items: the item IDs the user has rated
        ratings: the user's (normalized) ratings for those items
        other: the item-feature matrix
        reg: the regularization term
    Returns:
        the user-feature vector (equivalent to V in the current Cholesky code)
    """
    M = other[items, :]
    nf = other.shape[1]
    regI = torch.eye(nf, device=other.device) * reg
    MMT = M.T @ M
    A = MMT + regI * len(items)

    V = M.T @ ratings
    L, info = torch.linalg.cholesky_ex(A)
    if int(info):
        raise RuntimeError("error computing Cholesky decomposition (not symmetric?)")
    V = V.reshape(1, nf, 1)
    V = torch.cholesky_solve(V, L).reshape(nf)

    return V


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


class BiasedMF(MFPredictor):
    """
    Biased matrix factorization trained with alternating least squares :cite:p:`Zhou2008-bj`.  This
    is a prediction-oriented algorithm suitable for explicit feedback data, using the alternating
    least squares approach to compute :math:`P` and :math:`Q` to minimize the regularized squared
    reconstruction error of the ratings matrix.

    It provides two solvers for the optimization step (the `method` parameter):

    ``'cd'`` (the default)
        Coordinate descent :cite:p:`Takacs2011-ix`, adapted for a separately-trained bias model and
        to use weighted regularization as in the original ALS paper :cite:p:`Zhou2008-bj`.
    ``'cholesky'``
        The original ALS :cite:p:`Zhou2008-bj`, using Cholesky decomposition
        to solve for the optimized matrices.
    ``'lu'``:
        Deprecated alias for ``'cholskey'``

    See the base class :class:`.MFPredictor` for documentation on
    the estimated parameters you can extract from a trained model.

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train
        reg(float): the regularization factor; can also be a tuple ``(ureg, ireg)`` to
            specify separate user and item regularization terms.
        damping(float): damping factor for the underlying bias.
        bias(bool or :class:`Bias`): the bias model.  If ``True``, fits a :class:`Bias` with
            damping ``damping``.
        rng_spec:
            Random number generator or state (see :func:`seedbank.numpy_rng`).
        progress: a :func:`tqdm.tqdm`-compatible progress bar function
    """

    timer = None
    bias: Bias | None

    user_index_: pd.Index | None
    item_index_: pd.Index | None
    user_features_: torch.Tensor | None
    item_features_: torch.Tensor | None

    def __init__(
        self,
        features,
        *,
        iterations=20,
        reg=0.1,
        damping=5,
        bias=True,
        rng_spec=None,
        progress=None,
        save_user_features=True,
    ):
        self.features = features
        self.iterations = iterations
        self.regularization = reg
        self.damping = damping
        if bias is True:
            self.bias = Bias(damping=damping)
        else:
            self.bias = bias or None
        self.progress = progress if progress is not None else util.no_progress
        self.rng = numpy_rng(rng_spec)
        self.save_user_features = save_user_features

    def fit(self, ratings, **kwargs):
        """
        Run ALS to train a model.

        Args:
            ratings: the ratings data frame.

        Returns:
            The algorithm (for chaining).
        """
        util.check_env()
        self.timer = util.Stopwatch()

        for epoch, algo in enumerate(self.fit_iters(ratings, **kwargs)):
            pass  # we just need to do the iterations

        if self.user_features_ is not None:
            _logger.info(
                "trained model in %s (|P|=%f, |Q|=%f)",
                self.timer,
                torch.norm(self.user_features_, "fro"),
                torch.norm(self.item_features_, "fro"),
            )
        else:
            _logger.info(
                "trained model in %s (|Q|=%f)",
                self.timer,
                torch.norm(self.item_features_, "fro"),
            )

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
            _logger.info("[%s] fitting bias model", self.timer)
            self.bias.fit(ratings)

        current, uctx, ictx = self._initial_model(ratings)

        _logger.info(
            "[%s] training biased MF model with ALS for %d features", self.timer, self.features
        )
        for epoch, model in enumerate(self._train_iters(current, uctx, ictx)):
            self._save_params(model)
            yield self

    def _save_params(self, model):
        "Save the parameters into model attributes."
        self.item_index_ = model.items
        self.user_index_ = model.users
        self.item_features_ = model.item_matrix
        if self.save_user_features:
            self.user_features_ = model.user_matrix
        else:
            self.user_features_ = None

    def _initial_model(self, ratings):
        # transform ratings using offsets
        if self.bias:
            _logger.info("[%s] normalizing ratings", self.timer)
            ratings = self.bias.transform(ratings)

        "Initialize a model and build contexts."
        rmat, users, items = sparse_ratings(ratings, torch=True)
        n_users = len(users)
        n_items = len(items)

        _logger.debug("setting up contexts")
        trmat = rmat.transpose(0, 1).to_sparse_csr()

        _logger.debug("initializing item matrix")
        imat = self.rng.standard_normal((n_items, self.features))
        imat /= np.linalg.norm(imat, axis=1).reshape((n_items, 1))
        imat = torch.from_numpy(imat)
        # imat = torch.from_numpy(imat)
        _logger.debug("|Q|: %f", torch.norm(imat, "fro"))
        _logger.debug("initializing user matrix")
        umat = self.rng.standard_normal((n_users, self.features))
        umat /= np.linalg.norm(umat, axis=1).reshape((n_users, 1))
        umat = torch.from_numpy(umat)
        # umat = torch.from_numpy(umat)
        _logger.debug("|P|: %f", torch.norm(umat, "fro"))

        if False:
            _logger.info("training on CUDA")
            imat = imat.to("cuda")
            umat = umat.to("cuda")
            rmat = rmat.to("cuda")
            trmat = trmat.to("cuda")

        return PartialModel(users, items, umat, imat), rmat, trmat

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
        assert uctx.shape == (n_users, n_items)
        assert ictx.shape == (n_items, n_users)

        if isinstance(self.regularization, tuple):
            ureg, ireg = self.regularization
        else:
            ureg = ireg = self.regularization

        u_trainer = ALSCholeskyTrainer(uctx, current.user_matrix, current.item_matrix, ureg)
        i_trainer = ALSCholeskyTrainer(ictx, current.item_matrix, current.user_matrix, ireg)

        ctx = tmp.get_context("spawn")
        with ctx.Pool(32, _worker_init, (u_trainer, i_trainer)) as pool:
            for epoch in self.progress(range(self.iterations), desc="BiasedMF", leave=False):
                du = _train_parallel(pool, u_trainer, "left")
                # du = _train_sequential(u_trainer)
                _logger.debug("[%s] finished user epoch %d", self.timer, epoch)
                di = _train_parallel(pool, i_trainer, "right")
                # di = _train_sequential(i_trainer)
                _logger.debug("[%s] finished item epoch %d", self.timer, epoch)
                _logger.info(
                    "[%s] finished epoch %d (|ΔP|=%.3f, |ΔQ|=%.3f)", self.timer, epoch, du, di
                )
                yield current

    def predict_for_user(self, user, items, ratings=None):
        scores = None
        u_offset = None
        if ratings is not None and len(ratings) > 0:
            if self.bias:
                ratings, u_offset = self.bias.transform_user(ratings)

            ri_idxes = self.item_index_.get_indexer_for(ratings.index)
            ri_good = ri_idxes >= 0
            ri_it = ri_idxes[ri_good]
            ri_val = ratings.values[ri_good]

            # unpack regularization
            if isinstance(self.regularization, tuple):
                ureg, ireg = self.regularization
            else:
                ureg = self.regularization

            u_feat = _train_bias_row_cholesky(ri_it, ri_val, self.item_features_, ureg)
            scores = self.score_by_ids(user, items, u_feat)
        else:
            # look up user index
            scores = self.score_by_ids(user, items)

        if self.bias and ratings is not None and len(ratings) > 0:
            return self.bias.inverse_transform_user(user, scores, u_offset)
        elif self.bias:
            return self.bias.inverse_transform_user(user, scores)
        else:
            return scores

    def __str__(self):
        return "als.BiasedMF(features={}, regularization={})".format(
            self.features, self.regularization
        )


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
            _logger.info(
                "[%s] finished training model with %d features (|P|=%f, |Q|=%f)",
                self.timer,
                self.features,
                np.linalg.norm(self.user_features_, "fro"),
                np.linalg.norm(self.item_features_, "fro"),
            )
        else:
            _logger.info(
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

        _logger.info(
            "[%s] training implicit MF model with ALS for %d features", self.timer, self.features
        )
        _logger.info(
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
            _logger.debug("[%s] finished user epoch %d", self.timer, epoch)
            di = train(ictx, current.item_matrix, current.user_matrix, ireg)
            _logger.debug("[%s] finished item epoch %d", self.timer, epoch)
            _logger.info("[%s] finished epoch %d (|ΔP|=%.3f, |ΔQ|=%.3f)", self.timer, epoch, du, di)
            yield current

    def _initial_model(self, ratings):
        "Initialize a model and build contexts."

        if not self.use_ratings:
            ratings = ratings[["user", "item"]]

        rmat, users, items = sparse_ratings(ratings)
        n_users = len(users)
        n_items = len(items)

        _logger.debug("setting up contexts")
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
