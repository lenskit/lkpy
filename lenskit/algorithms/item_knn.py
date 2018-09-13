"""
Item-based k-NN collaborative filtering.
"""

from collections import namedtuple
import logging

import pandas as pd
import numpy as np
import scipy.sparse as sps

from lenskit import util, matrix
from . import _item_knn as accel
from . import Trainable, Predictor

_logger = logging.getLogger(__package__)

IIModel = namedtuple('IIModel', ['items', 'means', 'counts', 'sim_matrix', 'rating_matrix'])


class ItemItem(Trainable, Predictor):
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
        if self.max_neighbors is not None and self.max_neighbors < 1:
            self.max_neighbors = -1
        self.min_neighbors = min_nbrs
        if self.min_neighbors is not None and self.min_neighbors < 1:
            self.min_neighbors = 1
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
            norm = np.linalg.norm(xmc)
            if norm > 1.0e-10:
                return xmc / norm
            else:
                return xmc

        uir = ratings.set_index(['item', 'user']).rating
        uir = uir.groupby('item').transform(normalize)
        uir = uir.reset_index()
        assert uir.rating.notna().all()
        # now we have normalized vectors

        _logger.info('[%s] computing similarity matrix', watch)
        sim_matrix, items = self._cy_matrix(ratings, uir, watch)
        item_means = item_means.reindex(items)

        _logger.info('[%s] computed %d neighbor pairs', watch, sim_matrix.nnz)

        return IIModel(items, item_means.values, np.diff(sim_matrix.indptr),
                       sim_matrix, ratings.set_index(['user', 'item']).rating)

    def _cy_matrix(self, ratings, uir, watch):
        _logger.debug('[%s] preparing Cython data launch', watch)
        # the Cython implementation requires contiguous numeric IDs.
        # so let's make those
        rmat, user_idx, item_idx = matrix.sparse_ratings(uir)
        assert rmat.nnz == len(uir)
        n_items = len(item_idx)

        context = accel.BuildContext(rmat)

        _logger.debug('[%s] running accelerated matrix computations', watch)
        ndf = accel.sim_matrix(context, self.min_similarity,
                               self.save_neighbors
                               if self.save_neighbors and self.save_neighbors > 0
                               else -1)
        _logger.info('[%s] got neighborhoods for %d of %d items',
                     watch, ndf.item.nunique(), n_items)
        smat = sps.csr_matrix((ndf.similarity.values, (ndf.item.values, ndf.neighbor.values)),
                              shape=(n_items, n_items))

        # sort each matrix row by value
        for i in range(n_items):
            start = smat.indptr[i]
            end = smat.indptr[i+1]
            sorti = np.argsort(smat.data[start:end])
            tmp = smat.indices[sorti[::-1] + start]
            smat.indices[start:end] = tmp
            tmp = smat.data[sorti[::-1] + start]
            smat.data[start:end] = tmp

        # clean up neighborhoods
        return smat, item_idx

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
        _logger.debug('predicting %d items for user %s', len(items), user)
        if ratings is None:
            if user not in model.rating_matrix.index:
                return pd.Series(np.nan, index=items)
            ratings = model.rating_matrix.loc[user]

        # set up rating array
        # get rated item positions & limit to in-model items
        ri_pos = model.items.get_indexer(ratings.index)
        m_rates = ratings[ri_pos >= 0]
        ri_pos = ri_pos[ri_pos >= 0]
        rate_v = np.full(len(model.items), np.nan, dtype=np.float_)
        rate_v[ri_pos] = m_rates.values - model.means[ri_pos]
        _logger.debug('user %s: %d of %d rated items in model', user, len(ri_pos), len(ratings))
        assert np.sum(np.logical_not(np.isnan(rate_v))) == len(ri_pos)

        # set up item result vector
        # ipos will be an array of item indices
        i_pos = model.items.get_indexer(items)
        i_pos = i_pos[i_pos >= 0]
        _logger.debug('user %s: %d of %d requested items in model', user, len(i_pos), len(items))

        # scratch result array
        iscore = np.full(len(model.items), np.nan, dtype=np.float_)

        # now compute the predictions
        accel.predict(model.sim_matrix, len(model.items),
                      self.min_neighbors, self.max_neighbors,
                      rate_v, i_pos, iscore)

        nscored = np.sum(np.logical_not(np.isnan(iscore)))
        iscore += model.means
        assert np.sum(np.logical_not(np.isnan(iscore))) == nscored

        results = pd.Series(iscore, index=model.items)
        results = results[results.notna()]
        results = results.reindex(items, fill_value=np.nan)
        assert results.notna().sum() == nscored

        _logger.debug('user %s: predicted for %d of %d items',
                      user, results.notna().sum(), len(items))

        return results

    def save_model(self, model, file):
        _logger.info('saving I-I model to %s', file)
        with pd.HDFStore(file, 'w') as hdf:
            h5 = hdf._handle
            group = h5.create_group('/', 'ii_model')
            h5.create_array(group, 'items', model.items.values)
            h5.create_array(group, 'means', model.means)
            _logger.debug('saving matrix with %d entries (%d nnz)',
                          model.sim_matrix.nnz, np.sum(model.sim_matrix.data != 0))
            h5.create_array(group, 'col_ptrs', model.sim_matrix.indptr)
            h5.create_array(group, 'row_nums', model.sim_matrix.indices)
            h5.create_array(group, 'sim_values', model.sim_matrix.data)

            hdf['ratings'] = model.rating_matrix

    def load_model(self, file):
        _logger.info('loading I-I model from %s', file)
        with pd.HDFStore(file, 'r') as hdf:
            ratings = hdf['ratings']
            h5 = hdf._handle

            items = h5.get_node('/ii_model', 'items').read()
            items = pd.Index(items)
            means = h5.get_node('/ii_model', 'means').read()

            indptr = h5.get_node('/ii_model', 'col_ptrs').read()
            indices = h5.get_node('/ii_model', 'row_nums').read()
            values = h5.get_node('/ii_model', 'sim_values').read()
            _logger.debug('loading matrix with %d entries (%d nnz)',
                          len(values), np.sum(values != 0))
            assert np.all(values > self.min_similarity)

            matrix = sps.csr_matrix((values, indices, indptr))

            return IIModel(items, means, np.diff(indptr), matrix, ratings)

    def __str__(self):
        return 'ItemItem(nnbrs={}, msize={})'.format(self.max_neighbors, self.save_neighbors)
