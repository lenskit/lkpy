"""
User-based k-NN collaborative filtering.
"""

from sys import intern
import pathlib
import logging

import pandas as pd
import numpy as np
from scipy import stats

from .. import util, matrix
from . import Predictor

_logger = logging.getLogger(__name__)


class UserUser(Predictor):
    """
    User-user nearest-neighbor collaborative filtering with ratings. This user-user implementation
    is not terribly configurable; it hard-codes design decisions found to work well in the previous
    Java-based LensKit code.

    Attributes:
        user_index_(pandas.Index): User index.
        item_index_(pandas.Index): Item index.
        user_means_(numpy.ndarray): User mean ratings.
        rating_matrix_(matrix.CSR): Normalized user-item rating matrix.
        transpose_matrix_(matrix.CSR): Transposed un-normalized rating matrix.
    """
    AGG_SUM = intern('sum')
    AGG_WA = intern('weighted-average')

    def __init__(self, nnbrs, min_nbrs=1, min_sim=0, center=True, aggregate='weighted-average'):
        """
        Args:
            nnbrs(int):
                the maximum number of neighbors for scoring each item (``None`` for unlimited)
            min_nbrs(int): the minimum number of neighbors for scoring each item
            min_sim(double): minimum similarity threshold for considering a neighbor
            center(bool):
                whether to normalize (mean-center) rating vectors.  Turn this off when working
                with unary data and other data types that don't respond well to centering.
            aggregate:
                the type of aggregation to do. Can be ``weighted-average`` or ``sum``.
        """
        self.nnbrs = nnbrs
        self.min_nbrs = min_nbrs
        self.min_sim = min_sim
        self.center = center
        self.aggregate = intern(aggregate)

    def fit(self, ratings):
        """
        "Train" a user-user CF model.  This memorizes the rating data in a format that is usable
        for future computations.

        Args:
            ratings(pandas.DataFrame): (user, item, rating) data for collaborative filtering.

        Returns:
            UUModel: a memorized model for efficient user-based CF computation.
        """

        uir, users, items = matrix.sparse_ratings(ratings)

        # mean-center ratings
        if self.center:
            umeans = np.zeros(len(users))
            for u in range(uir.nrows):
                sp, ep = uir.row_extent(u)
                v = uir.values[sp:ep]
                umeans[u] = m = v.mean()
                uir.values[sp:ep] -= m
        else:
            umeans = None

        # compute centered transpose
        iur = uir.transpose()

        # L2-normalize ratings
        if uir.values is None:
            uir.values = np.full(uir.nnz, 1.0)
        for u in range(uir.nrows):
            sp, ep = uir.row_extent(u)
            v = uir.values[sp:ep]
            n = np.linalg.norm(v)
            uir.values[sp:ep] /= n

        mkl = matrix.mkl_ops()
        mkl_m = mkl.SparseM.from_csr(uir) if mkl else None

        self.rating_matrix_ = uir
        self.user_index_ = users
        self.user_means_ = umeans
        self.item_index_ = items
        self.transpose_matrix_ = iur
        self._mkl_m_ = mkl_m

        return self

    def predict_for_user(self, user, items, ratings=None):
        """
        Compute predictions for a user and items.

        Args:
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series):
                the user's ratings (indexed by item id); if provided, will be used to
                recompute the user's bias at prediction time.

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """

        watch = util.Stopwatch()
        items = pd.Index(items, name='item')

        ratings, umean = self._get_user_data(user, ratings)
        if ratings is None:
            return pd.Series(index=items)
        assert len(ratings) == len(self.item_index_)  # ratings is a dense vector

        # now ratings is normalized to be a mean-centered unit vector
        # this means we can dot product to score neighbors
        # score the neighbors!
        if self._mkl_m_:
            nsims = np.zeros(len(self.user_index_))
            nsims = self._mkl_m_.mult_vec(1, ratings, 0, nsims)
        else:
            rmat = self.rating_matrix_.to_scipy()
            nsims = rmat @ ratings
        assert len(nsims) == len(self.user_index_)
        if user in self.user_index_:
            nsims[self.user_index_.get_loc(user)] = 0

        _logger.debug('computed user similarities')

        results = np.full(len(items), np.nan, dtype=np.float_)
        ri_pos = self.item_index_.get_indexer(items.values)
        for i in range(len(results)):
            ipos = ri_pos[i]
            if ipos < 0:
                continue

            # get the item's users & ratings
            i_users = self.transpose_matrix_.row_cs(ipos)

            # find and limit the neighbors
            i_sims = nsims[i_users]
            mask = np.abs(i_sims >= 1.0e-10)

            if self.nnbrs is not None and self.nnbrs > 0:
                rank = stats.rankdata(-i_sims, 'ordinal')
                mask = np.logical_and(mask, rank <= self.nnbrs)
            if self.min_sim is not None:
                mask = np.logical_and(mask, i_sims >= self.min_sim)

            if np.sum(mask) < self.min_nbrs:
                continue

            # now we have picked weights, take a dot product
            ism = i_sims[mask]
            if self.aggregate == self.AGG_WA:
                i_rates = self.transpose_matrix_.row_vs(ipos)
                v = np.dot(i_rates[mask], ism)
                v = v / np.sum(ism)
            elif self.aggregate == self.AGG_SUM:
                v = np.sum(ism)
            else:
                raise ValueError('invalid aggregate ' + self.aggregate)
            results[i] = v + umean

        results = pd.Series(results, index=items, name='prediction')

        _logger.debug('scored %d of %d items for %s in %s',
                      results.notna().sum(), len(items), user, watch)
        return results

    def _get_user_data(self, user, ratings):
        "Get a user's data for user-user CF"
        rmat = self.rating_matrix_

        if ratings is None:
            try:
                upos = self.user_index_.get_loc(user)
                ratings = rmat.row(upos)
                umean = self.user_means_[upos] if self.user_means_ is not None else 0
            except KeyError:
                _logger.warning('user %d has no ratings and none provided', user)
                return None, 0
        else:
            _logger.debug('using provided ratings for user %d', user)
            if self.center:
                umean = ratings.mean()
                ratings = ratings - umean
            else:
                umean = 0
            unorm = np.linalg.norm(ratings)
            ratings = ratings / unorm
            ratings = ratings.reindex(self.item_index_, fill_value=0).values

        return ratings, umean

    def __getstate__(self):
        state = dict(self.__dict__)
        if '_mkl_m_' in state:
            del state['_mkl_m_']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.aggregate = intern(self.aggregate)
        mkl = matrix.mkl_ops()
        self._mkl_m_ = mkl.SparseM.from_csr(self.rating_matrix_) if mkl else None

    def __str__(self):
        return 'UserUser(nnbrs={}, min_sim={})'.format(self.nnbrs, self.min_sim)
