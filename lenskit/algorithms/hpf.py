import logging

import pandas as pd

from . import Predictor, Trainable
from .mf_common import MFModel

_logger = logging.getLogger(__name__)


class HPF(Predictor, Trainable):
    """
    Hierarchical Poisson factorization, provided by hpfrec_.

    .. _hpfrec: https://hpfrec.readthedocs.io/en/latest/

    Args:
        features(int): the number of features
        **kwargs: arguments passed to :py:class:`hpfrec.HPF`.
    """

    def __init__(self, features, **kwargs):
        self.features = features
        self._kwargs = kwargs

    def train(self, ratings):
        import hpfrec

        users = pd.Index(ratings.user.unique())
        items = pd.Index(ratings.item.unique())

        hpfdf = pd.DataFrame({
            'UserId': users.get_indexer(ratings.user),
            'ItemId': items.get_indexer(ratings.item),
            'Count': ratings.rating.values.copy()
        })

        hpf = hpfrec.HPF(self.features, reindex=False, **self._kwargs)

        _logger.info('fitting HPF model with %d features', self.features)
        hpf.fit(hpfdf)

        return MFModel(users, items, hpf.Theta, hpf.Beta)

    def predict(self, model: MFModel, user, items, ratings=None):
        # look up user index
        return model.score_by_ids(user, items)
