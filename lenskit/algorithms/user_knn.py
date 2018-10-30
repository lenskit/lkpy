"""
User-based k-NN collaborative filtering.
"""

from collections import namedtuple
import pathlib
import logging

import pandas as pd
import numpy as np
from scipy import stats

from .. import util, matrix
from . import Trainable, Predictor

_logger = logging.getLogger(__name__)

UUModel = namedtuple('UUModel', ['matrix', 'user_means', 'items', 'transpose', 'mkl_m'])
UUModel.__doc__ = """
Memorized data for user-user collaborative filtering.

Attributes:
    matrix(matrix.CSR): normalized user-item rating matrix.
    user_means: user mean ratings.
    items: index of item IDs
    transpose(matrix.CSR):
        the transposed rating matrix (with data transformations but without L2 normalization).
"""


class UserUser(Trainable, Predictor):
    """
    User-user nearest-neighbor collaborative filtering with ratings. This user-user implementation
    is not terribly configurable; it hard-codes design decisions found to work well in the previous
    Java-based LensKit code.
    """

    def __init__(self, nnbrs, min_nbrs=1, min_sim=0):
        """
        Args:
            nnbrs(int):
                the maximum number of neighbors for scoring each item (``None`` for unlimited)
            min_nbrs(int): the minimum number of neighbors for scoring each item
            min_sim(double): minimum similarity threshold for considering a neighbor
        """
        self.max_neighbors = nnbrs
        self.min_neighbors = min_nbrs
        self.min_similarity = min_sim

    def train(self, ratings):
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
        umeans = np.zeros(len(users))
        for u in range(uir.nrows):
            sp, ep = uir.row_extent(u)
            v = uir.values[sp:ep]
            umeans[u] = m = v.mean()
            uir.values[sp:ep] -= m

        # compute centered transpose
        iur = uir.transpose()

        # L2-normalize ratings
        for u in range(uir.nrows):
            sp, ep = uir.row_extent(u)
            v = uir.values[sp:ep]
            n = np.linalg.norm(v)
            uir.values[sp:ep] /= n

        umeans = pd.Series(umeans, index=users, name='mean')

        mkl = matrix.mkl_ops()
        mkl_m = mkl.SparseM.from_csr(uir) if mkl else None

        return UUModel(uir, umeans, items, iur, mkl_m)

    def predict(self, model, user, items, ratings=None):
        """
        Compute predictions for a user and items.

        Args:
            model (UUModel): the memorized data to use.
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series):
                the user's ratings (indexed by item id); if provided, will be used to
                recompute the user's bias at prediction time.

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """

        watch = util.Stopwatch()

        ratings, umean = self._get_user_data(model, user, ratings)
        if ratings is None:
            return pd.Series(index=items)
        assert len(ratings) == len(model.items)  # ratings is a dense vector

        rmat = model.matrix
        rmat = matrix.csr_to_scipy(rmat)

        # now ratings is normalized to be a mean-centered unit vector
        # this means we can dot product to score neighbors
        # score the neighbors!
        if model.mkl_m:
            nsims = np.zeros(len(model.user_means))
            nsims = model.mkl_m.mult_vec(1, ratings, 0, nsims)
        else:
            nsims = rmat @ ratings
        assert len(nsims) == len(model.user_means.index)
        if user in model.user_means.index:
            nsims[model.user_means.index.get_loc(user)] = 0

        _logger.debug('computed user similarities')

        results = pd.Series(np.nan, index=items, name='prediction')
        for i in range(len(results)):
            item = results.index[i]
            try:
                ipos = model.items.get_loc(item)
            except KeyError:
                continue

            # now we have the item, let us find it!
            iu_sp = model.transpose.rowptrs[ipos]
            iu_ep = model.transpose.rowptrs[ipos+1]

            # get its users & ratings
            i_users = model.transpose.colinds[iu_sp:iu_ep]
            i_rates = model.transpose.values[iu_sp:iu_ep]

            # find and limit the neighbors
            i_sims = nsims[i_users]
            mask = np.abs(i_sims >= 1.0e-10)

            if self.max_neighbors is not None and self.max_neighbors > 0:
                rank = stats.rankdata(-i_sims, 'ordinal')
                mask = np.logical_and(mask, rank <= self.max_neighbors)
            if self.min_similarity is not None:
                mask = np.logical_and(mask, i_sims >= self.min_similarity)

            if np.sum(mask) < self.min_neighbors:
                continue

            # now we have picked weights, take a dot product
            ism = i_sims[mask]
            v = np.dot(i_rates[mask], ism)
            v = v / np.sum(ism)
            results.iloc[i] = v + umean

        _logger.debug('scored %d of %d items for %s in %s',
                      results.notna().sum(), len(items), user, watch)
        return results

    def _get_user_data(self, model, user, ratings):
        "Get a user's data for user-user CF"
        rmat = model.matrix

        if ratings is None:
            if user not in model.user_means.index:
                _logger.warning('user %d has no ratings and none provided', user)
                return None, 0

            upos = model.user_means.index.get_loc(user)
            ratings = rmat.row(upos)
            umean = model.user_means.iloc[upos]
        else:
            _logger.debug('using provided ratings for user %d', user)
            umean = ratings.mean()
            ratings = ratings - umean
            unorm = np.linalg.norm(ratings)
            ratings = ratings / unorm
            ratings = ratings.reindex(model.items, fill_value=0).values

        return ratings, umean

    def save_model(self, model, path):
        path = pathlib.Path(path)
        _logger.info('saving to %s', path)

        m_rows = matrix.csr_rowinds(model.matrix)
        t_rows = matrix.csr_rowinds(model.transpose)

        np.savez_compressed(path, users=model.user_means.index.values, items=model.items,
                            user_means=model.user_means.values,
                            m_rows=m_rows, m_cols=model.matrix.colinds, m_vals=model.matrix.values,
                            t_rows=t_rows, t_cols=model.transpose.colinds,
                            t_vals=model.transpose.values)

    def load_model(self, path):
        path = util.npz_path(path)

        with np.load(path) as npz:
            users = npz['users']
            users = pd.Index(users, name='user')
            items = pd.Index(npz['items'], name='item')
            user_means = pd.Series(npz['user_means'], index=users, name='mean')
            mat = matrix.csr_from_coo(npz['m_rows'], npz['m_cols'], npz['m_vals'])
            tx = matrix.csr_from_coo(npz['t_rows'], npz['t_cols'], npz['t_vals'])

        mkl = matrix.mkl_ops()
        mkl_m = mkl.SparseM.from_csr(mat) if mkl else None
        return UUModel(mat, user_means, items, tx, mkl_m)

    def __str__(self):
        return 'UserUser(nnbrs={}, min_sim={})'.format(self.max_neighbors, self.min_similarity)
