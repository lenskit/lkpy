"""
Common utilities & implementations for matrix factorization.
"""

import pathlib
import logging

import numpy as np
import pandas as pd

from .. import check, util, sharing
from . import basic

_logger = logging.getLogger(__name__)


class MFModel(sharing.Shareable):
    """
    Common model for matrix factorization.

    Attributes:
        user_index(pandas.Index): Users in the model (length=:math:`m`).
        item_index(pandas.Index): Items in the model (length=:math:`n`).
        user_features(numpy.ndarray): The :math:`m \\times k` user-feature matrix.
        item_features(numpy.ndarray): The :math:`n \\times k` item-feature matrix.
    """

    def __init__(self, users, items, umat, imat):
        check.check_dimension(users, umat, 'user matrix', d2=0)
        check.check_dimension(items, imat, 'item matrix', d2=0)
        check.check_dimension(umat, imat, 'user & item matrices', 1, 1)

        self.user_index = users
        self.item_index = items
        self.user_features = umat
        self.item_features = imat

    @property
    def n_features(self):
        "The number of features."
        return self.user_features.shape[1]

    @property
    def n_users(self):
        "The number of users."
        return len(self.users)

    @property
    def n_items(self):
        "The number of items."
        return len(self.items)

    def lookup_user(self, user):
        """
        Look up the index for a user.

        Args:
            user: the user ID to look up

        Returns:
            int: the user index.
        """
        try:
            return self.user_index.get_loc(user)
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
        return self.item_index.get_indexer(items)

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
        uv = self.user_features[user, :]
        # get item matrix
        im = self.item_features[items, :]
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

    def share_publish(self, context):
        uik = context.put_index(self.user_index)
        iik = context.put_index(self.item_index)
        umk = context.put_array(self.user_features)
        imk = context.put_array(self.item_features)
        return uik, iik, umk, imk

    @classmethod
    def share_resolve(cls, key, context):
        uik, iik, umk, imk = key
        return cls(context.get_index(uik), context.get_index(iik),
                   context.get_array(umk), context.get_array(imk))

    def save(self, path):
        np.savez_compressed(path, users=self.user_index.values, items=self.item_index.values,
                            umat=self.user_features, imat=self.item_features)

    @classmethod
    def load(cls, path):
        path = util.npz_path(path)

        with np.load(path) as npz:
            users = pd.Index(npz['users'], name='user')
            items = pd.Index(npz['items'], name='item')
            umat = npz['umat']
            imat = npz['imat']
            return cls(users, items, umat, imat)


class BiasMFModel(MFModel):
    """
    Common model for biased matrix factorization.

    Attributes:
        user_index(pandas.Index): Users in the model (length=:math:`m`).
        item_index(pandas.Index): Items in the model (length=:math:`n`).
        global_bias(double): The global bias term.
        user_bias(numpy.ndarray): The user bias terms.
        item_bias(numpy.ndarray): The item bias terms.
        user_features(numpy.ndarray): The :math:`m \\times k` user-feature matrix.
        item_features(numpy.ndarray): The :math:`n \\times k` item-feature matrix.
    """

    def __init__(self, users, items, bias, umat, imat):
        super().__init__(users, items, umat, imat)
        if isinstance(bias, basic.BiasModel):
            self.global_bias = bias.mean
            self.user_bias = bias.users.reindex(users, fill_value=0).values
            self.item_bias = bias.items.reindex(items, fill_value=0).values
        else:
            self.global_bias, self.user_bias, self.item_bias = bias

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
            rv = rv + self.global_bias
            if self.user_bias is not None:
                rv = rv + self.user_bias[user]
            if self.item_bias is not None:
                rv = rv + self.item_bias[items]

        return rv

    def share_publish(self, context):
        uik = context.put_index(self.user_index)
        iik = context.put_index(self.item_index)
        ubk = context.put_array(self.user_bias)
        ibk = context.put_array(self.item_bias)
        umk = context.put_array(self.user_features)
        imk = context.put_array(self.item_features)
        return uik, iik, (self.global_bias, ubk, ibk), umk, imk

    @classmethod
    def share_resolve(cls, key, context):
        uik, iik, bias, umk, imk = key
        gbias, ubk, ibk = bias
        return cls(context.get_index(uik), context.get_index(iik),
                   (gbias, context.get_array(ubk), context.get_array(ibk)),
                   context.get_array(umk), context.get_array(imk))

    def save(self, path):
        np.savez_compressed(path, users=self.user_index.values, items=self.item_index.values,
                            umat=self.user_features, imat=self.item_features,
                            gbias=np.array([self.global_bias]),
                            ubias=self.user_bias, ibias=self.item_bias)

    @classmethod
    def load(cls, path: pathlib.Path):
        path = util.npz_path(path)

        with np.load(path) as npz:
            users = pd.Index(npz['users'])
            items = pd.Index(npz['items'])
            umat = npz['umat']
            imat = npz['imat']
            gbias = npz['gbias'][0]
            ubias = npz['ubias']
            ibias = npz['ibias']
            return cls(users, items, (gbias, ubias, ibias), umat, imat)
