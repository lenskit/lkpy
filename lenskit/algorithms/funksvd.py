"""
FunkSVD (biased MF).
"""

from collections import namedtuple
import logging

import pandas as pd
import numpy as np
import numba as n

from . import Trainable, Predictor
from . import basic

_logger = logging.getLogger(__package__)

BiasMFModel = namedtuple('BiasMFModel', ['user_index', 'item_index',
                                         'global_bias', 'user_bias', 'item_bias',
                                         'user_features', 'item_features'])


@n.jitclass([
    ('users', n.int32[:]),
    ('items', n.int32[:]),
    ('ratings', n.double[:]),
    ('bias', n.double[:]),
    ('n_samples', n.uint64)
])
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


@n.jitclass([
    ('iter_count', n.int32),
    ('learning_rate', n.double),
    ('reg_term', n.double),
    ('rmin', n.double),
    ('rmax', n.double)
])
class _Params:
    def __init__(self, niters, lrate, reg, rmin, rmax):
        self.iter_count = niters
        self.learning_rate = lrate
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


@n.jitclass([
    ('user_features', n.double[:, :]),
    ('item_features', n.double[:, :]),
    ('feature_count', n.int32),
    ('user_count', n.int32),
    ('item_count', n.int32),
    ('initial_value', n.double)
])
class Model:
    def __init__(self, umat, imat):
        self.user_features = umat
        self.item_features = imat
        self.feature_count = umat.shape[1]
        assert imat.shape[1] == self.feature_count
        self.user_count = umat.shape[0]
        self.item_count = imat.shape[0]
        self.initial_value = np.nan


def _fresh_model(nfeatures, nusers, nitems, init=0.1):
    umat = np.full([nusers, nfeatures], init, dtype=np.float_)
    imat = np.full([nitems, nfeatures], init, dtype=np.float_)
    model = Model(umat, imat)
    model.initial_value = init
    assert model.feature_count == nfeatures
    assert model.user_count == nusers
    assert model.item_count == nitems
    return model


@n.njit
def _train_feature(ctx, params, model, est, f, trail):
    users = ctx.users
    items = ctx.items
    ratings = ctx.ratings
    umat = model.user_features
    imat = model.item_features

    for epoch in range(params.iter_count):
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
            ufd = ufd * params.learning_rate
            acc_ud += ufd * ufd
            ifd = error * ufv - params.reg_term * ifv
            ifd = ifd * params.learning_rate
            acc_id += ifd * ifd
            umat[user, f] += ufd
            imat[item, f] += ifd

    return np.sqrt(sse / ctx.n_samples)


def train(ctx: Context, params: _Params, model: Model):
    est = ctx.bias

    for f in range(model.feature_count):
        trail = model.initial_value * model.initial_value * (model.feature_count - f - 1)
        rmse = _train_feature(ctx, params, model, est, f, trail)
        _logger.info('finished feature %d (RMSE=%f)', f, rmse)

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


class FunkSVD(Predictor, Trainable):
    """
    Algorithm class implementing FunkSVD matrix factorization.

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train each feature
        lrate(double): the learning rate
        reg(double): the regularization factor
        damping(double): damping factor for the underlying mean
        range(tuple):
            the ``(min, max)`` rating values to clamp ratings, or ``None`` to leave
            predictions unclamped.
    """

    def __init__(self, features, iterations=100, lrate=0.001, reg=0.015, damping=5, range=None):
        self.features = features
        self.iterations = iterations
        self.learning_rate = lrate
        self.regularization = reg
        self.damping = damping
        self.range = range

    def train(self, ratings, bias=None):
        """
        Train a FunkSVD model.

        Args:
            ratings: the ratings data frame.
            bias(.bias.BiasModel): a pre-trained bias model to use.

        Returns:
            The trained biased MF model.
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

        _logger.info('preparing rating data for %d samples', len(ratings))
        _logger.debug('shuffling rating data')
        shuf = np.arange(len(ratings), dtype=np.int_)
        np.random.shuffle(shuf)
        ratings = ratings.iloc[shuf, :]

        _logger.debug('indexing users and items')
        uidx = pd.Index(ratings.user.unique())
        iidx = pd.Index(ratings.item.unique())

        users = uidx.get_indexer(ratings.user).astype(np.int_)
        assert np.all(users >= 0)
        items = iidx.get_indexer(ratings.item).astype(np.int_)
        assert np.all(items >= 0)

        _logger.debug('computing initial estimates')
        initial = pd.Series(gbias, index=ratings.index, dtype=np.float_)
        ibias, initial = _align_add_bias(ibias, iidx, ratings.item, initial)
        ubias, initial = _align_add_bias(ubias, uidx, ratings.user, initial)

        _logger.debug('have %d estimates for %d ratings', len(initial), len(ratings))
        assert len(initial) == len(ratings)

        _logger.debug('initializing data structures')
        context = Context(users, items, ratings.rating.astype(np.float_).values,
                          initial.values)
        params = make_params(self.iterations, self.learning_rate, self.regularization, self.range)

        model = _fresh_model(self.features, len(uidx), len(iidx))

        _logger.info('training biased MF model with %d features', self.features)
        train(context, params, model)
        _logger.info('finished model training')

        return BiasMFModel(uidx, iidx, gbias, ubias, ibias,
                           model.user_features, model.item_features)

    def predict(self, model, user, items, ratings=None):
        if user not in model.user_index:
            return pd.Series(np.nan, index=items)

        # get user index
        uidx = model.user_index.get_loc(user)
        assert uidx >= 0

        # get item index & limit to valid ones
        items = np.array(items)
        iidx = model.item_index.get_indexer(items)
        good = iidx >= 0
        good_items = items[good]
        good_iidx = iidx[good]

        # get user vector
        uv = model.user_features[uidx, :]
        # get item matrix
        im = model.item_features[good_iidx, :]

        # multiply
        _logger.debug('scoring %d items for user %s', len(good_items), user)
        rv = np.matmul(im, uv)
        assert rv.shape[0] == len(good_items)
        assert len(rv.shape) == 1
        # add bias back in
        rv = rv + model.global_bias
        if model.user_bias is not None:
            rv = rv + model.user_bias.iloc[uidx]
        if model.item_bias is not None:
            rv = rv + model.item_bias.iloc[good_iidx].values

        # clamp if suitable
        if self.range is not None:
            rmin, rmax = self.range
            rv = np.maximum(rv, rmin)
            rv = np.minimum(rv, rmax)

        res = pd.Series(rv, index=good_items)
        res = res.reindex(items)
        return res
