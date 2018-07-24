"""
FunkSVD (biased MF).
"""

from collections import namedtuple
import logging
import warnings

import pandas as pd
import numpy as np

from .. import util as lku
from .. import check
from . import basic
from . import _funksvd as _fsvd

_logger = logging.getLogger(__package__)

BiasMFModel = namedtuple('BiasMFModel', ['user_index', 'item_index',
                                         'global_bias', 'user_bias', 'item_bias',
                                         'user_features', 'item_features'])


class FunkSVD:
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
            ibias = ibias.loc[ratings.item]
            ibias.index = ratings.index
            initial = initial + ibias
        if ubias is not None:
            ubias = ubias.reindex(uidx, fill_value=0)
            assert len(ubias) == len(uidx)
            ubias = ubias.loc[ratings.user]
            ubias.index = ratings.index
            initial = initial + ubias
        _logger.debug('have %d estimates for %d ratings', len(initial), len(ratings))
        assert len(initial) == len(ratings)

        _logger.debug('initializing data structures')
        context = _fsvd.Context(users, items, ratings.rating.values, initial.values)
        params = _fsvd.Params(self.iterations, self.learning_rate, self.regularization)

        model = _fsvd.Model(self.features, len(uidx), len(iidx))

        _logger.info('training biased MF model with %d features', self.features)
        if self.range is None:
            _fsvd.train_unclamped(context, params, model)
        else:
            min, max = self.range
            _fsvd.train_clamped(context, umat, imat,
                                self.iterations, self.learning_rate, self.regularization, min, max)

        return BiasMFModel(uidx, iidx, gbias, ubias, ibias, model.user_features, model.item_features)
