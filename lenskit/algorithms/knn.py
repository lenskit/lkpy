"""
k-NN collaborative filtering.
"""

from collections import namedtuple
import logging
import time
from line_profiler import LineProfiler

import pandas as pd
import numpy as np
import scipy as sp

from lenskit import util

_logger = logging.getLogger(__package__)

UUModel = namedtuple('UUModel', ['matrix', 'user_stats', 'item_users'])


class UserUser:
    """
    User-user nearest-neighbor collaborative filtering.  This user-user implementation is not
    terribly configurable; it hard-codes design decisions found to work well in the previous
    Java-based LensKit code.
    """

    def __init__(self, nnbrs, min_nbrs=1, min_sim=0):
        self.max_neighbors = nnbrs
        self.min_neighbors = min_nbrs
        self.min_similarity = min_sim

    def train(self, ratings):
        """
        "Train" a user-user CF model.  This memorizes the rating data in a format that is usable
        for future computations.
        """

        user_means = ratings.groupby('user').rating.mean()
        uir = ratings.set_index(['user', 'item']).rating
        uir -= user_means
        unorms = uir.groupby('user').agg(np.linalg.norm)
        uir /= unorms
        ustats = pd.DataFrame({'mean': user_means, 'norm': unorms})
        iusers = ratings.set_index('item').user

        return UUModel(uir.reset_index(name='rating'), ustats, iusers)

    @profile
    def predict(self, model, user, items, ratings=None):
        """
        Compute predictions for a user and items.

        Args:
            model (BiasModel): the trained model to use.
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series):
                the user's ratings (indexed by item id); if provided, will be used to
                recompute the user's bias at prediction time.

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """

        watch = util.Stopwatch()
        rmat = model.matrix

        if ratings is None:
            if user not in model.user_stats.index:
                _logger.warn('user %d has no ratings and none provided', user)
                return pd.Series()
            ratings = rmat[rmat.user == user].set_index('item').rating
            umean = model.user_stats.loc[user, 'mean']
            unorm = model.user_stats.loc[user, 'norm']
        else:
            _logger.debug('using provided ratings for user %d', user)
            umean = ratings.mean()
            ratings = ratings - umean
            unorm = np.linalg.norm(ratings)
            ratings = ratings / unorm

        assert ratings.index.names == ['item']

        # now ratings is normalized to be a mean-centered unit vector
        # this means we can dot product to score neighbors
        # let's find all users who have rated one of our items
        # get rid of
        kitems = pd.Series(items)
        kitems = kitems[kitems.isin(model.item_users.index)]
        candidates = model.item_users.loc[kitems].unique()
        # don't use ourselves to predict
        candidates = candidates[candidates != user]
        candidates = pd.Index(candidates)
        # and get all ratings by them, and for one of the rated items
        # nbr_ratings = model.matrix.loc[(candidates, slice(None))].reset_index()
        nbr_ratings = rmat[rmat.user.isin(candidates)]
        nbr_ratings = nbr_ratings[nbr_ratings.item.isin(ratings.index)]
        _logger.debug('predicting for user %d with %d candidate users and %d ratings',
                      user, len(candidates), len(nbr_ratings))

        # we can dot-product to compute similarities
        @profile
        def sim(df):
            nbr = df.set_index('item').rating
            nbr, mine = nbr.align(ratings, join='inner')
            return nbr.dot(mine)
        nbr_sims = nbr_ratings.groupby('user', sort=False).apply(sim)
        assert nbr_sims.index.name == 'user'
        # filter for similarity threshold
        nbr_sims = nbr_sims[nbr_sims > self.min_similarity]
        nbr_sims.name = 'similarity'
        _logger.debug('have %d possible users after similarity filtering', len(nbr_sims))

        # now that we have the final similarities, we are ready to compute predictions
        # grab the neighbor ratings for all target items
        nbr_tgt_rates = rmat[rmat.user.isin(nbr_sims.index) & rmat.item.isin(items)]
        nbr_tgt_rates.set_index(['user', 'item'], inplace=True)

        # add in our user similarities
        pred_f = nbr_tgt_rates.join(nbr_sims)

        @profile
        def score(idf):
            if len(idf) < self.min_neighbors:
                return np.nan
            sims = idf.similarity
            rates = idf.rating * model.user_stats['norm']
            if self.max_neighbors is not None:
                sims = sims.nlargest(self.max_neighbors)
                sims, rates = sims.align(rates, join='inner')
            return sims.dot(rates) / sims.abs().sum() + umean

        results = pred_f.groupby('item').apply(score)
        _logger.info('scored %d of %d items for %s in %s',
                     results.notna().sum(), len(items), user, watch)
        return results
