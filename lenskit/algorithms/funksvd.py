"""
FunkSVD (biased MF).
"""

from collections import namedtuple
import logging
import warnings

import pandas as pd
import numpy as np

from . import Trainable, Predictor
from .. import util as lku
from .. import check
from . import basic
from . import _funksvd as _fsvd

_logger = logging.getLogger(__package__)

BiasMFModel = namedtuple('BiasMFModel', ['user_index', 'item_index',
                                         'global_bias', 'user_bias', 'item_bias',
                                         'user_features', 'item_features'])


class FunkSVD(Predictor, Trainable):
    def __init__(self, features, iterations=100, lrate=0.001, reg=0.02, damping=5, range=None):
        self.features = features
        self.iterations = iterations
        self.learning_rate = lrate
        self.regularization = reg
        self.damping = damping
        self.range = range

    def train(self, ratings, bias=None):
        """
        Train a FunkSVD model.
        """
        if bias is None:
            _logger.info('training bias model')
            bias = basic.Bias(damping=self.damping).train(ratings)
        # unpack the bias
        if isinstance(bias, basic.BiasModel):
            gbias = bias.mean
            ibias = bias.items
            ubias = bias.items
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

        users = uidx.get_indexer(ratings.user)
        items = iidx.get_indexer(ratings.item)

        _logger.debug('computing initial estimates')
        initial = pd.Series(gbias, index=ratings.index, dtype=np.float_)
        if ibias is not None:
            # realign ibias to make sure it matches
            ibias = ibias.reindex(iidx, fill_value=0)
            assert len(ibias) == len(iidx)
            ibs = ibias.loc[ratings.item]
            ibs.index = ratings.index
            initial = initial + ibs
        if ubias is not None:
            ubias = ubias.reindex(uidx, fill_value=0)
            assert len(ubias) == len(uidx)
            ubs = ubias.loc[ratings.user]
            ubs.index = ratings.index
            initial = initial + ubs
        _logger.debug('have %d estimates for %d ratings', len(initial), len(ratings))
        assert len(initial) == len(ratings)

        _logger.debug('initializing data structures')
        context = _fsvd.Context(users, items, ratings.rating.astype(np.float_).values,
                                initial.values)
        params = _fsvd.Params(self.iterations, self.learning_rate, self.regularization)

        model = _fsvd.Model.fresh(self.features, len(uidx), len(iidx))

        _logger.info('training biased MF model with %d features', self.features)
        _fsvd.train(context, params, model, self.range)
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
        rv = rv + model.global_bias
        if model.user_bias is not None:
            assert model.user_bias.index[uidx] == user
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
