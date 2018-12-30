import logging

import pandas as pd

from .mf_common import MFPredictor

_logger = logging.getLogger(__name__)


class HPF(MFPredictor):
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

    def fit(self, ratings):
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

        self.user_index_ = users
        self.item_index_ = items
        self.user_features_ = hpf.Theta
        self.item_features_ = hpf.Beta

        return self

    def predict_for_user(self, user, items, ratings=None):
        # look up user index
        return self.score_by_ids(user, items)
