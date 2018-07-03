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
from . import baselines

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
            bias = baselines.Bias(damping=self.damping).train(ratings)
        # unpack the bias
        if isinstance(bias, baselines.BiasModel):
            gbias = bias.mean
            ibias = bias.items
            ubias = bias.items
        else:
            # we have a single global bias (for e.g. implicit feedback data)
            gbias = bias
            ibias = None
            ubias = None

        uidx = pd.Index(ratings.user.unique())
        iidx = pd.Index(ratings.item.unique())

        users = uidx.get_indexer(ratings.user)
        items = iidx.get_indexer(ratings.item)
        initial = pd.Series(gbias, index=ratings.index, dtype=np.float_)
        if ibias is not None:
            # realign ibias to make sure it matches
            ibias = ibias.reindex(iidx, fill_value=0)
            initial = initial + ibias[ratings.item]
        if ubias is not None:
            ubias = ubias.reindex(uidx, fill_value=0)
            initial = initial + ubias[ratings.user]

        umat = np.full([len(uidx), self.features], 0.1, dtype=np.float_)
        imat = np.full([len(iidx), self.features], 0.1, dtype=np.float_)

        if self.range is None:
            _fsvd.train_unclamped(users, items, ratings, initial, umat, imat,
                                  self.iterations, self.learning_rate, self.regularization)
        else:
            min, max = self.range
            _fsvd.train_clamped(users, items, ratings, initial, umat, imat,
                                self.iterations, self.learning_rate, self.regularization, min, max)

        return BiasMFModel(uidx, iidx, gbias, ubias, ibias, umat, imat)
