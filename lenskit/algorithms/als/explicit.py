from __future__ import annotations

import logging
import math
from typing import Optional, cast

import numpy as np
import pandas as pd
import torch
from progress_api import make_progress
from seedbank import SeedLike, numpy_rng

from lenskit.parallel.config import ensure_parallel_init

from ... import util
from ...data import sparse_ratings
from ..bias import Bias
from ..mf_common import MFPredictor
from .common import PartialModel, TrainContext

_log = logging.getLogger(__name__)


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
        features: the number of features to train
        epochs: the number of iterations to train
        reg: the regularization factor; can also be a tuple ``(ureg, ireg)`` to
            specify separate user and item regularization terms.
        damping: damping factor for the underlying bias.
        bias: the bias model.  If ``True``, fits a :class:`Bias` with
            damping ``damping``.
        rng_spec:
            Random number generator or state (see :func:`seedbank.numpy_rng`).
        progress: a :func:`tqdm.tqdm`-compatible progress bar function
    """

    timer = None

    features: int
    epochs: int
    reg: float | tuple[float, float]
    bias: Bias | None
    rng: np.random.Generator
    save_user_features: bool

    user_index_: pd.Index | None
    item_index_: pd.Index
    user_features_: torch.Tensor | None
    item_features_: torch.Tensor

    def __init__(
        self,
        features: int,
        *,
        epochs: int = 20,
        reg: float | tuple[float, float] = 0.1,
        damping: float = 5,
        bias: bool | Bias = True,
        rng_spec: Optional[SeedLike] = None,
        save_user_features: bool = True,
    ):
        self.features = features
        self.epochs = epochs
        self.reg = reg
        self.damping = damping
        if bias is True:
            self.bias = Bias(damping=damping)
        else:
            self.bias = bias or None
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
        ensure_parallel_init()
        self.timer = util.Stopwatch()

        for algo in self.fit_iters(ratings, **kwargs):
            pass  # we just need to do the iterations

        if self.user_features_ is not None:
            _log.info(
                "trained model in %s (|P|=%f, |Q|=%f)",
                self.timer,
                torch.norm(self.user_features_, "fro"),
                torch.norm(self.item_features_, "fro"),
            )
        else:
            _log.info(
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
            _log.info("[%s] fitting bias model", self.timer)
            self.bias.fit(ratings)

        current, uctx, ictx = self._initial_model(ratings)

        _log.info(
            "[%s] training biased MF model with ALS for %d features", self.timer, self.features
        )
        for model in self._train_iters(current, uctx, ictx):
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
            _log.info("[%s] normalizing ratings", self.timer)
            ratings = self.bias.transform(ratings)

        "Initialize a model and build contexts."
        rmat, users, items = sparse_ratings(ratings, torch=True)
        n_users = len(users)
        n_items = len(items)

        _log.debug("setting up contexts")
        trmat = rmat.transpose(0, 1).to_sparse_csr()

        _log.debug("initializing item matrix")
        imat = self.rng.standard_normal((n_items, self.features))
        imat /= np.linalg.norm(imat, axis=1).reshape((n_items, 1))
        imat = torch.from_numpy(imat)
        _log.debug("|Q|: %f", torch.norm(imat, "fro"))

        _log.debug("initializing user matrix")
        umat = self.rng.standard_normal((n_users, self.features))
        umat /= np.linalg.norm(umat, axis=1).reshape((n_users, 1))
        umat = torch.from_numpy(umat)
        _log.debug("|P|: %f", torch.norm(umat, "fro"))

        if False:
            _log.info("training on CUDA")
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

        if isinstance(self.reg, tuple):
            ureg, ireg = self.reg
        else:
            ureg = ireg = self.reg

        with make_progress("BiasedMF", self.epochs) as epb:
            for epoch in range(self.epochs):
                # du = _train_parallel(pool, u_trainer, "left")
                # du = _train_sequential(u_trainer)
                du = _train_matrix_cholesky(uctx, current.user_matrix, current.item_matrix, ureg)
                _log.debug("[%s] finished user epoch %d", self.timer, epoch)
                # di = _train_parallel(pool, i_trainer, "right")
                # di = _train_sequential(i_trainer)
                di = _train_matrix_cholesky(ictx, current.item_matrix, current.user_matrix, ireg)
                _log.debug("[%s] finished item epoch %d", self.timer, epoch)
                _log.info(
                    "[%s] finished epoch %d (|ΔP|=%.3f, |ΔQ|=%.3f)", self.timer, epoch, du, di
                )
                epb.update()
                yield current

    def predict_for_user(self, user, items, ratings: Optional[pd.Series] = None):
        scores = None
        u_offset = None
        if ratings is not None and len(ratings) > 0:
            if self.bias:
                ratings, u_offset = self.bias.transform_user(ratings)
            ratings = cast(pd.Series, ratings)

            ri_idxes = self.item_index_.get_indexer_for(ratings.index)
            ri_good = ri_idxes >= 0
            ri_it = torch.from_numpy(ri_idxes[ri_good])
            ri_val = torch.from_numpy(ratings.values[ri_good])

            # unpack regularization
            if isinstance(self.reg, tuple):
                ureg, ireg = self.reg
            else:
                ureg = self.reg

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
        return "als.BiasedMF(features={}, regularization={})".format(self.features, self.reg)


@torch.jit.script
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


@torch.jit.script
def _train_update_rows(ctx: TrainContext, start: int, end: int) -> torch.Tensor:
    result = ctx.left[start:end, :].clone()

    for i in range(start, end):
        row = ctx.matrix[i]
        (n,) = row.shape
        if n == 0:
            continue

        cols = row.indices()[0]
        vals = row.values().type(ctx.left.type())

        V = _train_solve_row(cols, vals, ctx.left, ctx.right, ctx.regI)
        result[i - start] = V

    return result


@torch.jit.script
def _train_update_fanout(ctx: TrainContext) -> float:
    if ctx.nrows <= 50:
        # at 50 rows, we run sequentially
        M = _train_update_rows(ctx, 0, ctx.nrows)
        sqerr = torch.norm(ctx.left - M)
        ctx.left[:, :] = M
        return sqerr.item()

    # no more than 1024 chunks, and chunks must be at least 20
    csize = max(ctx.nrows // 1024, 20)

    results: list[tuple[int, int, torch.jit.Future[torch.Tensor]]] = []
    for start in range(0, ctx.nrows, csize):
        end = min(start + csize, ctx.nrows)
        results.append((start, end, torch.jit.fork(_train_update_rows, ctx, start, end)))

    sqerr = torch.tensor(0.0)
    for start, end, r in results:
        M = r.wait()
        diff = (ctx.left[start:end, :] - M).ravel()
        sqerr += torch.dot(diff, diff)
        ctx.left[start:end, :] = M

    return sqerr.item()


def _train_matrix_cholesky(
    mat: torch.Tensor, this: torch.Tensor, other: torch.Tensor, reg: float
) -> float:
    """
    One half of an explicit ALS training round using Cholesky decomposition on the normal
    matrices to solve the least squares problem.

    Args:
        mat: the :math:`m \\times n` matrix of ratings
        this: the :math:`m \\times k` matrix to train
        other: the :math:`n \\times k` matrix of sample features
        reg: the regularization term
    """
    context = TrainContext.create(mat, this, other, reg)

    sqerr = _train_update_fanout(context)

    return math.sqrt(sqerr)


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
