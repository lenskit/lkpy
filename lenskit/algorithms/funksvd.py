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
        _fsvd.train(context, params, model, self._kernel)

        return BiasMFModel(uidx, iidx, gbias, ubias, ibias,
                           model.user_features, model.item_features)

    def predict(self, model, user, items, ratings):
        uidx = model.user_index.get_loc(user)
        iidx = model.item_index.get_indexer(items)
        kern = self._kernel
        m = _fsvd.Model(model.user_features, model.item_features)

        ubase = model.global_bias
        if model.user_bias is not None:
            ubase += model.user_bias.iloc[0]

        result = pd.Series(ubase, index=items)
        for i in range(len(iidx)):
            ii = iidx[i]
            if ii >= 0:
                ibase = ubase
                if model.item_bias is not None:
                    ibase += model.item_bias.loc[items[i]]
                result.iloc[ii] = kern.score(m, uidx, ii, ibase)


    @property
    def _kernel(self):
        if self.range is None:
            return _fsvd.DotKernel()
        else:
            min, max = self.range
            return _fsvd.ClampKernel(min, max)
