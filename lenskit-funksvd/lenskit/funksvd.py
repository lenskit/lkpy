# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
FunkSVD (biased MF).
"""

import logging
import time

import numba as n
import numpy as np
from numba.experimental import jitclass
from seedbank import numpy_rng

from lenskit import util
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.mf_common import MFPredictor
from lenskit.data import Dataset

_logger = logging.getLogger(__name__)


@jitclass(
    [
        ("user_features", n.float64[:, :]),
        ("item_features", n.float64[:, :]),
        ("feature_count", n.int32),
        ("user_count", n.int32),
        ("item_count", n.int32),
        ("initial_value", n.float64),
    ]
)  # type: ignore
class Model:
    "Internal model class for training SGD MF."

    def __init__(self, umat, imat):
        self.user_features = umat
        self.item_features = imat
        self.feature_count = umat.shape[1]
        assert imat.shape[1] == self.feature_count
        self.user_count = umat.shape[0]
        self.item_count = imat.shape[0]
        self.initial_value = np.nan


def _fresh_model(nfeatures, nusers, nitems, init=0.1):
    umat = np.full([nusers, nfeatures], init, dtype=np.float64)
    imat = np.full([nitems, nfeatures], init, dtype=np.float64)
    model = Model(umat, imat)
    model.initial_value = init
    assert model.feature_count == nfeatures
    assert model.user_count == nusers
    assert model.item_count == nitems
    return model


@jitclass(
    [
        ("iter_count", n.int32),
        ("lrate", n.float64),
        ("reg_term", n.float64),
        ("rmin", n.float64),
        ("rmax", n.float64),
    ]
)  # type: ignore
class _Params:
    def __init__(self, niters, lrate, reg, rmin, rmax):
        self.iter_count = niters
        self.lrate = lrate
        self.reg_term = reg
        self.rmin = rmin
        self.rmax = rmax


def make_params(niters, lrate, reg, range):
    if range is None:
        rmin = -np.inf
        rmax = np.inf
    else:
        rmin, rmax = range

    return _Params(niters, lrate, reg, rmin, rmax)


@jitclass([("est", n.float64[:]), ("feature", n.int32), ("trail", n.float64)])  # type: ignore
class _FeatContext:
    def __init__(self, est, feature, trail):
        self.est = est
        self.feature = feature
        self.trail = trail


@jitclass(
    [
        ("users", n.int32[:]),
        ("items", n.int32[:]),
        ("ratings", n.float64[:]),
        ("bias", n.float64[:]),
        ("n_samples", n.uint64),
    ]
)  # type: ignore
class Context:
    def __init__(self, users, items, ratings, bias):
        self.users = users
        self.items = items
        self.ratings = ratings
        self.bias = bias
        self.n_samples = users.shape[0]

        assert items.shape[0] == self.n_samples
        assert ratings.shape[0] == self.n_samples
        assert bias.shape[0] == self.n_samples


@n.njit
def _feature_loop(ctx: Context, params: _Params, model: Model, fc: _FeatContext):
    users = ctx.users
    items = ctx.items
    ratings = ctx.ratings
    umat = model.user_features
    imat = model.item_features
    est = fc.est
    f = fc.feature
    trail = fc.trail

    sse = 0.0
    acc_ud = 0.0
    acc_id = 0.0
    for s in range(ctx.n_samples):
        user = users[s]
        item = items[s]
        ufv = umat[user, f]
        ifv = imat[item, f]

        pred = est[s] + ufv * ifv + trail
        if pred < params.rmin:
            pred = params.rmin
        elif pred > params.rmax:
            pred = params.rmax

        error = ratings[s] - pred
        sse += error * error

        # compute deltas
        ufd = error * ifv - params.reg_term * ufv
        ufd = ufd * params.lrate
        acc_ud += ufd * ufd
        ifd = error * ufv - params.reg_term * ifv
        ifd = ifd * params.lrate
        acc_id += ifd * ifd
        umat[user, f] += ufd
        imat[item, f] += ifd

    return np.sqrt(sse / ctx.n_samples)


@n.njit
def _train_feature(ctx, params, model, fc):
    for epoch in range(params.iter_count):
        rmse = _feature_loop(ctx, params, model, fc)

    return rmse


def train(ctx: Context, params: _Params, model: Model, timer):
    est = ctx.bias

    for f in range(model.feature_count):
        start = time.perf_counter()
        trail = model.initial_value * model.initial_value * (model.feature_count - f - 1)
        fc = _FeatContext(est, f, trail)
        rmse = _train_feature(ctx, params, model, fc)
        end = time.perf_counter()
        _logger.info("[%s] finished feature %d (RMSE=%f) in %.2fs", timer, f, rmse, end - start)

        est = est + model.user_features[ctx.users, f] * model.item_features[ctx.items, f]
        est = np.maximum(est, params.rmin)
        est = np.minimum(est, params.rmax)


def _align_add_bias(bias, index, keys, series):
    "Realign a bias series with an index, and add to a series"
    # realign bias to make sure it matches
    bias = bias.reindex(index, fill_value=0)
    assert len(bias) == len(index)
    # look up bias for each key
    ibs = bias.loc[keys]
    # change index
    ibs.index = keys.index
    # and add
    series = series + ibs
    return bias, series


class FunkSVD(MFPredictor[np.ndarray]):
    """
    Algorithm class implementing FunkSVD matrix factorization.  FunkSVD is a regularized
    biased matrix factorization technique trained with featurewise stochastic gradient
    descent.

    See the base class :class:`.MFPredictor` for documentation on the estimated parameters
    you can extract from a trained model.

    Args:
        features: the number of features to train
        iterations: the number of iterations to train each feature
        lrate: the learning rate
        reg: the regularization factor
        damping: damping factor for the underlying mean
        bias: the underlying bias model to fit.  If ``True``, then a
            :py:class:`.bias.Bias` model is fit with ``damping``.
        range:
            the ``(min, max)`` rating values to clamp ratings, or ``None`` to leave
            predictions unclamped.
        random_state:
            The random state for shuffling the data prior to training.
    """

    features: int
    iterations: int
    lrate: float
    reg: float
    damping: float | tuple[float, float]
    range: tuple[float, float] | None
    bias: Bias | None
    random: np.random.Generator

    def __init__(
        self,
        features: int,
        iterations: int = 100,
        *,
        lrate: float = 0.001,
        reg: float = 0.015,
        damping: float | tuple[float, float] = 5,
        range: tuple[float, float] | None = None,
        bias: bool | Bias | None = True,
        random_state=None,
    ):
        self.features = features
        self.iterations = iterations
        self.lrate = lrate
        self.reg = reg
        self.damping = damping
        self.range = range
        if not bias:
            bias = None
        if bias is True:
            self.bias = Bias(damping=damping)
        else:
            self.bias = bias
        self.random = numpy_rng(random_state)

    def fit(self, data: Dataset, **kwargs):
        """
        Train a FunkSVD model.

        Args:
            ratings: the ratings data frame.
        """
        timer = util.Stopwatch()
        rate_df = data.interaction_matrix(format="pandas", layout="coo", field="rating")

        if self.bias:
            _logger.info("[%s] fitting bias model", timer)
            self.bias.fit(data)

        _logger.info("[%s] preparing rating data for %d samples", timer, len(rate_df))
        _logger.debug("shuffling rating data")
        shuf = np.arange(len(rate_df), dtype=np.int_)
        self.random.shuffle(shuf)
        rate_df = rate_df.iloc[shuf, :]

        users = np.array(rate_df["user_num"])
        items = np.array(rate_df["item_num"])
        ratings = np.array(rate_df["rating"], dtype=np.float64)

        _logger.debug("[%s] computing initial estimates", timer)
        if self.bias:
            initial = np.full(len(users), self.bias.mean_, dtype=np.float64)
            if self.bias.item_offsets_ is not None:
                initial += self.bias.item_offsets_.values[items]
            if self.bias.user_offsets_ is not None:
                initial += self.bias.user_offsets_.values[users]
        else:
            initial = np.zeros(len(users))

        _logger.debug("have %d estimates for %d ratings", len(initial), len(rate_df))
        assert len(initial) == len(rate_df)

        _logger.debug("[%s] initializing data structures", timer)
        context = Context(users, items, ratings, initial)
        params = make_params(self.iterations, self.lrate, self.reg, self.range)

        model = _fresh_model(self.features, data.users.size, data.items.size)

        _logger.info("[%s] training biased MF model with %d features", timer, self.features)
        train(context, params, model, timer)
        _logger.info("finished model training in %s", timer)

        self.users_ = data.users.copy()
        self.items_ = data.items.copy()
        self.user_features_ = model.user_features
        self.item_features_ = model.item_features

        return self

    def predict_for_user(self, user, items, ratings=None):
        # look up user index
        # look up user index
        preds = self.score_by_ids(user, items)
        if self.bias is not None:
            preds = self.bias.inverse_transform_user(user, preds)

        # clamp if suitable
        if self.range is not None:
            rmin, rmax = self.range
            preds = np.clip(preds, rmin, rmax)

        return preds

    def __str__(self):
        return "FunkSVD(features={}, reg={})".format(self.features, self.reg)
