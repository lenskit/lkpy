"""
User-based k-NN collaborative filtering.
"""

from collections import namedtuple
import logging

import pandas as pd
import numpy as np

from .. import util
from . import Trainable, Predictor

_logger = logging.getLogger(__package__)

UUModel = namedtuple('UUModel', ['matrix', 'user_stats', 'item_users'])
UUModel.__doc__ = "Memorized data for user-user collaborative filtering."
UUModel.matrix.__doc__ = \
    "(user, item, rating) data, where user vectors are mean-centered and unit-normalized."
UUModel.user_stats.__doc__ = \
    """
    (user, mean, norm) data, where mean is the user's raw mean rating and norm is the L2 norm
    of their mean-centered rating vector.  Together with ``matrix``, this can reconstruct the
    original rating matrix.
    """
UUModel.item_users.__doc__ = \
    """
    A series, indexed by item ID, of the user IDs who have rated each item.
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

        user_means = ratings.groupby('user').rating.mean()
        uir = ratings.set_index(['user', 'item']).rating
        uir -= user_means
        unorms = uir.groupby('user').agg(np.linalg.norm)
        uir /= unorms
        ustats = pd.DataFrame({'mean': user_means, 'norm': unorms})
        iusers = ratings.set_index('item').user

        return UUModel(uir.reset_index(name='rating'), ustats, iusers)

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

        ratings, umean, unorm = self._get_user_data(model, user, ratings)
        if ratings is None:
            return pd.Series(index=items)

        assert ratings.index.names == ['item']
        rmat = model.matrix

        # now ratings is normalized to be a mean-centered unit vector
        # this means we can dot product to score neighbors
        # get the candidate neighbors with their ratings for user-rated items
        nbr_ratings = self._get_nbr_ratings(model, user, items, ratings)

        nbr_sims = self._compute_similarities(nbr_ratings, ratings)
        _logger.debug('have %d possible users after similarity filtering', len(nbr_sims))

        # now that we have the final similarities, we are ready to compute predictions
        # grab the neighbor ratings for all target items
        nbr_tgt_rates = rmat[rmat.user.isin(nbr_sims.index) & rmat.item.isin(items)]
        nbr_tgt_rates.set_index(['user', 'item'], inplace=True)

        # add in our user similarities
        pred_f = nbr_tgt_rates.join(nbr_sims)

        # inner function for computing scores
        def score(idf):
            if len(idf) < self.min_neighbors:
                return np.nan

            if self.max_neighbors is not None:
                idf = idf.nlargest(self.max_neighbors, 'similarity')

            sims = idf.similarity
            rates = idf.rating * model.user_stats['norm']
            return sims.dot(rates) / sims.abs().sum() + umean

        # compute each item's score
        results = pred_f.groupby('item').apply(score)
        _logger.debug('scored %d of %d items for %s in %s',
                      results.notna().sum(), len(items), user, watch)
        return results

    def _get_user_data(self, model, user, ratings):
        "Get a user's data for user-user CF"
        rmat = model.matrix

        if ratings is None:
            if user not in model.user_stats.index:
                _logger.warn('user %d has no ratings and none provided', user)
                return None, 0, 0
            ratings = rmat[rmat.user == user].set_index('item').rating
            umean = model.user_stats.loc[user, 'mean']
            unorm = model.user_stats.loc[user, 'norm']
        else:
            _logger.debug('using provided ratings for user %d', user)
            umean = ratings.mean()
            ratings = ratings - umean
            unorm = np.linalg.norm(ratings)
            ratings = ratings / unorm

        return ratings, umean, unorm

    def _get_nbr_ratings(self, model, user, items, ratings):
        rmat = model.matrix
        # let's find all users who have rated one of our target items
        kitems = pd.Series(items)
        kitems = kitems[kitems.isin(model.item_users.index)]
        candidates = model.item_users.loc[kitems].unique()
        # don't use ourselves to predict
        candidates = candidates[candidates != user]
        candidates = pd.Index(candidates)
        # and get all ratings by them candidates, and for one of our rated items
        # this is the basis for computing similarities
        nbr_ratings = rmat[rmat.user.isin(candidates)]
        nbr_ratings = nbr_ratings[nbr_ratings.item.isin(ratings.index)]
        _logger.debug('predicting for user %d with %d candidate users and %d ratings',
                      user, len(candidates), len(nbr_ratings))

        return nbr_ratings

    def _compute_similarities(self, nbr_ratings, ratings):
        # we can dot-product to compute similarities
        # set up neighbor ratings to join with our ratings
        nr2 = nbr_ratings.set_index('item')
        nr2 = nr2.join(pd.DataFrame({'my_rating': ratings}))
        # compute product & sum by user (bulk dot product0)
        nr2['rp'] = nr2.rating * nr2.my_rating
        nbr_sims = nr2.groupby('user', sort=False).rp.sum()
        assert nbr_sims.index.name == 'user'
        # filter for similarity threshold
        nbr_sims = nbr_sims[nbr_sims >= self.min_similarity]
        nbr_sims.name = 'similarity'
        return nbr_sims

    def save_model(self, model, file):
        with pd.HDFStore(file, 'w') as store:
            store['matrix'] = model.matrix
            store['stats'] = model.user_stats
            store['item_users'] = model.item_users

    def load_model(self, file):
        with pd.HDFStore(file, 'r') as store:
            return UUModel(store['matrix'], store['stats'], store['item_users'])
