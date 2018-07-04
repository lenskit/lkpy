"""
Item-based k-NN collaborative filtering.
"""

from collections import namedtuple
import logging

import pandas as pd
import numpy as np

from lenskit import util
from . import _item_knn as accel

_logger = logging.getLogger(__package__)

IIModel = namedtuple('IIModel', ['item_means', 'sim_matrix', 'rating_matrix'])


class ItemItem:
    """
    Item-item nearest-neighbor collaborative filtering with ratings. This item-item implementation
    is not terribly configurable; it hard-codes design decisions found to work well in the previous
    Java-based LensKit code.
    """

    def __init__(self, nnbrs, min_nbrs=1, min_sim=1.0e-6, save_nbrs=None):
        """
        Args:
            nnbrs(int):
                the maximum number of neighbors for scoring each item (``None`` for unlimited)
            min_nbrs(int): the minimum number of neighbors for scoring each item
            min_sim(double): minimum similarity threshold for considering a neighbor
            save_nbrs(double):
                the number of neighbors to save per item in the trained model
                (``None`` for unlimited)
        """
        self.max_neighbors = nnbrs
        self.min_neighbors = min_nbrs
        self.min_similarity = min_sim
        self.save_neighbors = save_nbrs

    def train(self, ratings):
        """
        Train a model.

        The model-training process depends on ``save_nbrs`` and ``min_sim``, but *not* on other
        algorithm parameters.

        Args:
            ratings(pandas.DataFrame):
                (user,item,rating) data for computing item similarities.

        Returns:
            a trained item-item CF model.
        """
        # Training proceeds in 2 steps:
        # 1. Normalize item vectors to be mean-centered and unit-normalized
        # 2. Compute similarities with pairwise dot products
        watch = util.Stopwatch()
        item_means = ratings.groupby('item').rating.mean()
        _logger.info('[%s] computed means for %d items', watch, len(item_means))

        _logger.info('[%s] normalizing user-item ratings', watch)

        def normalize(x):
            xmc = x - x.mean()
            return xmc / np.linalg.norm(xmc)

        uir = ratings.set_index(['item', 'user']).rating
        uir = uir.groupby('item').transform(normalize)
        uir = uir.reset_index()
        # now we have normalized vectors

        _logger.info('[%s] computing simialrity matrix', watch)
        neighborhoods = self._cy_matrix(ratings, uir, watch)

        _logger.info('[%s] computed %d neighbor pairs', watch, len(neighborhoods))
        return IIModel(item_means, neighborhoods, ratings.set_index(['user', 'item']).rating)

    def _cy_matrix(self, ratings, uir, watch):
        _logger.debug('[%s] preparing Cython data launch', watch)
        # the Cython implementation requires contiguous numeric IDs.
        # so let's make those
        user_idx = pd.Index(ratings.user.unique())
        item_idx = pd.Index(ratings.item.unique())

        # convert input user/item cols to positions, & retain old indices for alignment
        t_users = user_idx.get_indexer(ratings.user)
        t_items = item_idx.get_indexer(ratings.item)
        assert len(t_users) == len(ratings)
        assert len(t_items) == len(ratings)

        # now we need the item-user grid: for each user, what are its items?
        iupairs = pd.DataFrame({'user': t_users, 'item': t_items}).sort_values(['item', 'user'])
        iu_items = iupairs.item.values
        iu_users = iupairs.user.values

        # now we need the item-user grid: for each item, what are its users & ratings?
        # _must_ be sorted by user
        uitriples = pd.DataFrame({'user': t_users, 'item': t_items,
                                  'rating': ratings.rating.values})
        uitriples = uitriples.sort_values(['user', 'item'])
        ui_users = uitriples.user.values
        ui_items = uitriples.item.values
        ui_ratings = uitriples.rating.values
        if ui_ratings.dtype != np.float_:
            ui_ratings = np.array(ui_ratings, dtype=np.float_)

        _logger.debug('[%s] running accelerated matrix computations', watch)
        neighborhoods = accel.sim_matrix(len(user_idx), len(item_idx),
                                         iu_items, iu_users, ui_users, ui_items, ui_ratings,
                                         self.min_similarity,
                                         self.save_neighbors if self.save_neighbors else -1)
        _logger.info('[%s] got neighborhoods for %d items', watch, neighborhoods.item.nunique())
        neighborhoods['item'] = item_idx[neighborhoods.item]
        neighborhoods['neighbor'] = item_idx[neighborhoods.neighbor]
        # clean up neighborhoods
        return neighborhoods.set_index('item')

    def _py_matrix(self, ratings, uir, watch):
        _logger.info('[%s] computing item-item similarities for %d items with %d ratings',
                     watch, uir.item.nunique(), len(uir))

        def sim_row(irdf):
            _logger.debug('[%s] computing similarities with %d ratings',
                          watch, len(irdf))
            assert irdf.index.name == 'user'
            # idf is all ratings for an item
            # join with other users' ratings
            # drop the item index, it's irrelevant
            irdf = irdf.rename(columns={'rating': 'tgt_rating', 'item': 'tgt_item'})
            # join with other ratings
            joined = irdf.join(uir, on='user', how='inner')
            assert joined.index.name == 'user'
            joined = joined[joined.tgt_item != joined.item]
            _logger.debug('[%s] using %d neighboring ratings to compute similarity',
                          watch, len(joined))
            # multiply ratings - dot product part 1
            joined['rp'] = joined.tgt_rating * joined.rating
            # group by item and sum
            sims = joined.groupby('item').rp.sum()
            if self.min_similarity is not None:
                sims = sims[sims >= self.min_similarity]
            if self.save_neighbors is not None:
                sims = sims.nlargest(self.save_neighbors)
            return sims.reset_index(name='similarity')\
                .rename(columns={'item': 'neighbor'})\
                .loc[:, ['neighbor', 'similarity']]

        neighborhoods = uir.groupby('item', sort=False).apply(sim_row)
        # get rid of extra groupby index
        neighborhoods = neighborhoods.reset_index(level=1, drop=True)
        return neighborhoods

    def predict(self, model, user, items, ratings=None):
        if ratings is None:
            ratings = model.rating_matrix.loc[user]
        ratings -= model.item_means

        # get the usable neighborhoods
        nbrhoods = model.sim_matrix.loc[items]
        nbrhoods = nbrhoods[nbrhoods.neighbor.isin(ratings.index)]
        if self.max_neighbors is not None:
            # rank neighbors & pick the best
            ranks = nbrhoods.groupby('item').similarity.rank(method='first', ascending=False)
            nbrhoods = nbrhoods[ranks <= self.max_neighbors]

        nbr_flip = nbrhoods.reset_index().set_index('neighbor')
        results = nbr_flip.groupby('item')\
            .apply(lambda idf: (idf.similarity * ratings).sum() / idf.similarity.sum())
        assert results.index.name == 'item'
        results += model.item_means
        # FIXME this is broken
        return results
