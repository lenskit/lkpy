from __future__ import annotations

import logging
import math
import warnings
from typing import Literal, Optional, TypeAlias

import numpy as np
import pandas as pd
import torch
from csr import CSR
from numba import njit, prange
from progress_api import make_progress
from seedbank import SeedLike, numpy_rng

from ... import util
from ...data import sparse_ratings
from ...math.solve import _dposv
from ..bias import Bias
from ..mf_common import MFPredictor
from .common import PartialModel, TrainContext

_log = logging.getLogger(__name__)
SolveMethod: TypeAlias = Literal["cg", "lu", "cholesky"]


class ImplicitMF(MFPredictor):
    """
    Implicit matrix factorization trained with alternating least squares
    :cite:p:`Hu2008-li`.  This algorithm outputs 'predictions', but they are not
    on a meaningful scale.  If its input data contains ``rating`` values, these
    will be used as the 'confidence' values; otherwise, confidence will be 1 for
    every rated item.

    See the base class :class:`.MFPredictor` for documentation on the estimated
    parameters you can extract from a trained model.

    With weight :math:`w`, this function decomposes the matrix
    :math:`\\mathbb{1}^* + Rw`, where :math:`\\mathbb{1}^*` is an :math:`m
    \\times n` matrix of all 1s.

    .. versionchanged:: 0.14
        By default, ``ImplicitMF`` ignores a ``rating`` column if one is present in the training
        data.  This can be changed through the ``use_ratings`` option.

    .. versionchanged:: 0.13
        In versions prior to 0.13, ``ImplicitMF`` used the rating column if it was present.
        In 0.13, we added an option to control whether or not the rating column is used; it
        initially defaulted to ``True``, but with a warning.  In 0.14 it defaults to ``False``.

    Args:
        features:
            The number of features to train
        epochs:
            The number of iterations to train
        reg:
            The regularization factor
        weight:
            The scaling weight for positive samples (:math:`\\alpha` in
            :cite:p:`Hu2008-li`).
        use_ratings:
            Whether to use the `rating` column, if present.  Defaults to
            ``False``; when ``True``, the values from the ``rating`` column are
            used, and multipled by ``weight``; if ``False``, ImplicitMF treats
            every rated user-item pair as having a rating of 1.
        method:
            the training method.

            ``'cg'`` (the default)
                Conjugate gradient method :cite:p:`Takacs2011-ix`.
            ``'cholesky'``
                A direct implementation of the original implicit-feedback ALS
                concept :cite:p:`Hu2008-li` using Cholesky decomposition to
                solve for the optimized matrices.

        rng_spec:
            Random number generator or state (see
            :func:`lenskit.util.random.rng`).
        progress: a :func:`tqdm.tqdm`-compatible progress bar function
    """

    timer = None

    features: int
    epochs: int
    reg: float | tuple[float, float]
    weight: float
    use_ratings: bool
    method: SolveMethod
    rng: np.random.Generator
    save_user_features: bool

    user_index_: pd.Index | None
    item_index_: pd.Index
    user_features_: torch.Tensor | None
    item_features_: torch.Tensor
    OtOr_: torch.Tensor

    def __init__(
        self,
        features: int,
        *,
        epochs: int = 20,
        reg: float | tuple[float, float] = 0.1,
        weight: float = 40,
        use_ratings: bool = False,
        method: SolveMethod = "cg",
        rng_spec: Optional[SeedLike] = None,
        save_user_features: bool = True,
    ):
        self.features = features
        self.epochs = epochs
        self.reg = reg
        self.weight = weight
        self.use_ratings = use_ratings
        if method == "lu":
            warnings.warn("method=lu has been renamed to cholesky")
            self.method = "cholesky"
        else:
            self.method = method
        self.rng = numpy_rng(rng_spec)
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
                torch.norm(self.user_features_, "fro"),
                torch.norm(self.item_features_, "fro"),
            )
        else:
            _log.info(
                "[%s] finished training model with %d features (|Q|=%f)",
                self.timer,
                self.features,
                torch.norm(self.item_features_, "fro"),
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
        for model in self._train_iters(current, uctx, ictx):
            self.save_params(model)
            yield self

    def save_params(self, model):
        self.item_index_ = model.items
        self.user_index_ = model.users
        self.item_features_ = model.item_matrix
        if self.save_user_features:
            self.user_features_ = model.user_matrix
        else:
            self.user_features_ = None

    def _initial_model(self, ratings):
        "Initialize a model and build contexts."

        if not self.use_ratings:
            ratings = ratings[["user", "item"]]

        rmat, users, items = sparse_ratings(ratings, torch=True)
        n_users = len(users)
        n_items = len(items)

        _log.debug("setting up contexts")
        rmat.values().mul_(self.weight)
        trmat = rmat.transpose(0, 1).to_sparse_csr()

        imat = self.rng.standard_normal((n_items, self.features)) * 0.01
        imat = np.square(imat)
        imat = torch.from_numpy(imat)

        umat = self.rng.standard_normal((n_users, self.features)) * 0.01
        umat = np.square(umat)
        umat = torch.from_numpy(umat)

        return PartialModel(users, items, umat, imat), rmat, trmat

    def _train_iters(self, current, uctx, ictx):
        "Generator of training iterations."
        if self.method == "cholesky":
            train = _train_implicit_cholesky
        elif self.method == "cg":
            train = _train_implicit_cg
        else:
            raise ValueError("unknown solver " + self.method)

        if isinstance(self.reg, tuple):
            ureg, ireg = self.reg
        else:
            ureg = ireg = self.reg

        with make_progress(_log, "ImplicitMF", self.epochs) as epb:
            for epoch in range(self.epochs):
                du = train(uctx, current.user_matrix, current.item_matrix, ureg)
                _log.debug("[%s] finished user epoch %d", self.timer, epoch)
                di = train(ictx, current.item_matrix, current.user_matrix, ireg)
                _log.debug("[%s] finished item epoch %d", self.timer, epoch)
                _log.info(
                    "[%s] finished epoch %d (|ΔP|=%.3f, |ΔQ|=%.3f)", self.timer, epoch, du, di
                )
                epb.update()
                yield current

    def predict_for_user(self, user, items, ratings: Optional[pd.Series] = None):
        if ratings is not None and len(ratings) > 0:
            ri_idxes = self.item_index_.get_indexer_for(ratings.index)
            ri_good = ri_idxes >= 0
            ri_it = ri_idxes[ri_good]
            if self.use_ratings is False:
                ri_val = np.ones(len(ri_good))
            else:
                ri_val = ratings.values[ri_good]
            ri_val *= self.weight

            ri_it = torch.from_numpy(ri_it)
            ri_val = torch.from_numpy(ri_val)

            u_feat = _train_implicit_row_cholesky(ri_it, ri_val, self.item_features_, self.OtOr_)
            return self.score_by_ids(user, items, u_feat)
        else:
            # look up user index
            return self.score_by_ids(user, items)

    def __str__(self):
        return "als.ImplicitMF(features={}, reg={}, w={})".format(
            self.features, self.reg, self.weight
        )


@torch.jit.script
def _cg_a_mult(
    OtOr: torch.Tensor, X: torch.Tensor, y: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    Compute the multiplication Av, where A = X'X + X'yX + λ.
    """
    XtXv = OtOr @ v
    XtyXv = X.T @ (y * (X @ v))
    return XtXv + XtyXv


@torch.jit.script
def _cg_solve(
    OtOr: torch.Tensor, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, epochs: int
) -> None:
    """
    Use conjugate gradient method to solve the system M†(X'X + X'yX + λ)w = M†X'(y+1).
    The parameter OtOr = X'X + λ.
    """
    nf = X.shape[1]
    # compute inverse of the Jacobi preconditioner
    Ad = torch.diag(OtOr).clone()
    for i in range(X.shape[0]):
        for k in range(nf):
            Ad[k] += X[i, k] * y[i] * X[i, k]

    iM = torch.reciprocal(Ad)

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
        gam = torch.dot(r, z)
        Ap = _cg_a_mult(OtOr, X, y, p)
        al = float(gam / torch.dot(p, Ap))
        w.add_(p, alpha=al)
        r.add_(Ap, alpha=-al)
        z = iM * r
        bet = torch.dot(r, z) / gam
        p = z + bet * p


@torch.jit.script
def _implicit_otor(other: torch.Tensor, reg: float) -> torch.Tensor:
    nf = other.shape[1]
    regmat = torch.eye(nf)
    regmat *= reg
    Ot = other.T
    OtO = Ot @ other
    OtO += regmat
    return OtO


@torch.jit.script
def _train_implicit_cg(
    mat: torch.Tensor, this: torch.Tensor, other: torch.Tensor, reg: float
) -> float:
    "One half of an implicit ALS training round with conjugate gradient."
    nr = mat.shape[0]

    OtOr = _implicit_otor(other, reg)

    frob = 0.0

    for i in range(nr):
        row = mat[i]
        (n,) = row.shape
        if n == 0:
            continue

        cols = row.indices()[0]
        vals = row.values().type(this.dtype)

        # we can optimize by only considering the nonzero entries of Cu-I
        # this means we only need the corresponding matrix columns
        M = other[cols, :]
        # and solve
        w = this[i, :].clone()
        _cg_solve(OtOr, M, vals, w, 3)

        # update stats
        delta = this[i, :] - w
        frob += torch.dot(delta, delta)

        # put back the result
        this[i, :] = w

    return math.sqrt(frob)


def _train_implicit_cholesky_rows(
    ctx: TrainContext, OtOr: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    result = ctx.left[start:end, :].clone()
    nf = ctx.left.shape[1]

    for i in range(start, end):
        row = ctx.matrix[i]
        (n,) = row.shape
        if n == 0:
            continue

        cols = row.indices()[0]
        vals = row.values().type(ctx.left.type())

        # we can optimize by only considering the nonzero entries of Cu-I
        # this means we only need the corresponding matrix columns
        M = ctx.right[cols, :]
        # Compute M^T (C_u-I) M, restricted to these nonzero entries
        MMT = (M.T * vals) @ M
        # assert MMT.shape[0] == ctx.n_features
        # assert MMT.shape[1] == ctx.n_features
        # Build the matrix for solving
        A = OtOr + MMT
        # Compute RHS - only used columns (p_ui != 0) values needed
        # Cu is rates + 1 for the cols, so just trim Ot
        y = ctx.right.T[:, cols] @ (vals + 1.0)
        # and solve
        L, info = torch.linalg.cholesky_ex(A)
        if int(info):
            raise RuntimeError("error computing Cholesky decomposition (not symmetric?)")
        y = y.reshape(1, nf, 1)
        y = torch.cholesky_solve(y, L).reshape(nf)
        # assert len(uv) == ctx.n_features
        result[i - start, :] = y

    return result


@torch.jit.script
def _train_implicit_cholesky_fanout(ctx: TrainContext, OtOr: torch.Tensor) -> float:
    if ctx.nrows <= 50:
        # at 50 rows, we run sequentially
        M = _train_implicit_cholesky_rows(ctx, OtOr, 0, ctx.nrows)
        sqerr = torch.norm(ctx.left - M)
        ctx.left[:, :] = M
        return sqerr.item()

    # no more than 1024 chunks, and chunk size must be at least 20
    csize = max(ctx.nrows // 1024, 20)

    results: list[tuple[int, int, torch.jit.Future[torch.Tensor]]] = []
    for start in range(0, ctx.nrows, csize):
        end = min(start + csize, ctx.nrows)
        results.append(
            (start, end, torch.jit.fork(_train_implicit_cholesky_rows, ctx, OtOr, start, end))
        )

    sqerr = torch.tensor(0.0)
    for start, end, r in results:
        M = r.wait()
        diff = (ctx.left[start:end, :] - M).ravel()
        sqerr += torch.dot(diff, diff)
        ctx.left[start:end, :] = M

    return sqerr.item()


def _train_implicit_cholesky(
    mat: torch.Tensor, this: torch.Tensor, other: torch.Tensor, reg: float
) -> float:
    "One half of an implicit ALS training round."
    context = TrainContext.create(mat, this, other, reg)
    OtOr = _implicit_otor(other, reg)

    sqerr = _train_implicit_cholesky_fanout(context, OtOr)

    return math.sqrt(sqerr)


@torch.jit.script
def _train_implicit_row_cholesky(
    items: torch.Tensor, ratings: torch.Tensor, other: torch.Tensor, otOr: torch.Tensor
) -> torch.Tensor:
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
    nf = other.shape[1]
    # Compute M^T (C_u-I) M, restricted to these nonzero entries
    MMT = (M.T * ratings) @ M
    # Build the matrix for solving
    A = otOr + MMT
    # Compute RHS - only used columns (p_ui != 0) values needed
    # Cu is rates + 1 for the cols, so just trim Ot
    y = other.T[:, items] @ (ratings + 1.0)
    # and solve
    L, info = torch.linalg.cholesky_ex(A)
    if int(info):
        raise RuntimeError("error computing Cholesky decomposition (not symmetric?)")
    y = y.reshape(1, nf, 1)
    y = torch.cholesky_solve(y, L).reshape(nf)

    return y
