"""
Common utilities & implementations for matrix factorization.
"""

import pathlib
import logging

import numpy as np
import pandas as pd

from .. import util
from . import Predictor

_logger = logging.getLogger(__name__)


class MFPredictor(Predictor):
    """
    Common predictor for matrix factorization.

    Attributes:
        user_index_(pandas.Index): Users in the model (length=:math:`m`).
        item_index_(pandas.Index): Items in the model (length=:math:`n`).
        user_features_(numpy.ndarray): The :math:`m \\times k` user-feature matrix.
        item_features_(numpy.ndarray): The :math:`n \\times k` item-feature matrix.
    """

    @property
    def n_features(self):
        "The number of features."
        return self.user_features_.shape[1]

    @property
    def n_users(self):
        "The number of users."
        return len(self.user_index_)

    @property
    def n_items(self):
        "The number of items."
        return len(self.item_index_)

    def lookup_user(self, user):
        """
        Look up the index for a user.

        Args:
            user: the user ID to look up

        Returns:
            int: the user index.
        """
        try:
            return self.user_index_.get_loc(user)
        except KeyError:
            return -1

    def lookup_items(self, items):
        """
        Look up the indices for a set of items.

        Args:
            items(array-like): the item IDs to look up.

        Returns:
            numpy.ndarray: the item indices. Unknown items will have negative indices.
        """
        return self.item_index_.get_indexer(items)

    def score(self, user, items):
        """
        Score a set of items for a user. User and item parameters must be indices
        into the matrices.

        Args:
            user(int): the user index
            items(array-like of int): the item indices
            raw(bool): if ``True``, do return raw scores without biases added back.

        Returns:
            numpy.ndarray: the scores for the items.
        """

        # get user vector
        uv = self.user_features_[user, :]
        # get item matrix
        im = self.item_features_[items, :]
        rv = np.matmul(im, uv)
        assert rv.shape[0] == len(items)
        assert len(rv.shape) == 1

        return rv

    def score_by_ids(self, user, items):
        uidx = self.lookup_user(user)
        if uidx < 0:
            _logger.debug('user %s not in model', user)
            return pd.Series(np.nan, index=items)

        # get item index & limit to valid ones
        items = np.array(items)
        iidx = self.lookup_items(items)
        good = iidx >= 0
        good_items = items[good]
        good_iidx = iidx[good]

        # multiply
        _logger.debug('scoring %d items for user %s', len(good_items), user)
        rv = self.score(uidx, good_iidx)

        res = pd.Series(rv, index=good_items)
        res = res.reindex(items)
        return res


class BiasMFPredictor(MFPredictor):
    """
    Common model for biased matrix factorization.

    Attributes:
        user_index_(pandas.Index): Users in the model (length=:math:`m`).
        item_index_(pandas.Index): Items in the model (length=:math:`n`).
        global_bias_(double): The global bias term.
        user_bias_(numpy.ndarray): The user bias terms.
        item_bias_(numpy.ndarray): The item bias terms.
        user_features_(numpy.ndarray): The :math:`m \\times k` user-feature matrix.
        item_features_(numpy.ndarray): The :math:`n \\times k` item-feature matrix.
    """

    def score(self, user, items, raw=False):
        """
        Score a set of items for a user. User and item parameters must be indices
        into the matrices.

        Args:
            user(int): the user index
            items(array-like of int): the item indices
            raw(bool): if ``True``, do return raw scores without biases added back.

        Returns:
            numpy.ndarray: the scores for the items.
        """

        rv = super().score(user, items)

        if not raw:
            # add bias back in
            rv = rv + self.global_bias_
            if self.user_bias_ is not None:
                rv = rv + self.user_bias_[user]
            if self.item_bias_ is not None:
                rv = rv + self.item_bias_[items]

        return rv
