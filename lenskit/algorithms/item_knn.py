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
        if self.max_neighbors is not None and self.max_neighbors < 0:
            self.max_neighbors = 0
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
            if xmc.abs().sum() > 1.0e-10:
                return xmc / np.linalg.norm(xmc)
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
        return IIModel(items, item_means, np.diff(sim_matrix.indptr),
                       sim_matrix, ratings.set_index(['user', 'item']).rating)

    def _cy_matrix(self, ratings, uir, watch):
        _logger.debug('[%s] preparing Cython data launch', watch)
        # the Cython implementation requires contiguous numeric IDs.
        # so let's make those
        rmat, user_idx, item_idx = matrix.sparse_ratings(uir)
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
        if ratings is None:
            ratings = model.rating_matrix.loc[user]

        # set up item series & results
        items = pd.Series(items)
        results = pd.Series(np.nan, index=items, dtype='f8')
        # get item positions
        positions = model.items.get_indexer(items)
        # reduce to items (and positions) in the model
        m_items = items.values[positions >= 0]
        m_pos = positions[positions >= 0]
        # reduce to items (and positions) with neighborhoods
        have_nbrs = model.counts[m_pos] > 0
        m_items = m_items[have_nbrs]
        m_pos = m_pos[have_nbrs]

        # same for ratings
        r_idx = model.items.get_indexer(ratings.index)
        m_rates = ratings[r_idx >= 0]
        m_rates -= model.means
        r_idx = r_idx[r_idx >= 0]
        # update index to be position-based, not id-based
        m_ratev = np.full(len(model.items), 0, dtype='f8')
        m_ratev[r_idx] = m_rates.values
        _logger.debug('user %s: %d of %d ratings in index', user, len(m_rates), len(ratings))

        # now compute each prediction
        for i in range(len(m_items)):
            item = m_items[i]
            ipos = m_pos[i]
            row = model.sim_matrix.getrow(ipos)
            nbr_mask = np.isin(row.indices, r_idx)
            nbr_is = row.indices[nbr_mask]
            nbr_vs = row.data[nbr_mask]
            if self.min_neighbors and len(nbr_is) < self.min_neighbors:
                continue
            if self.max_neighbors and len(nbr_is) > self.max_neighbors:
                # slice, since we sorted the row by value in training
                nbr_is = nbr_is[:self.max_neighbors]
                nbr_vs = nbr_vs[:self.max_neighbors]

            # look up ratings by position!
            n_rates = m_ratev[nbr_is]
            assert len(n_rates) == len(nbr_vs)
            weights = nbr_vs
            if self.min_similarity is None or self.min_similarity < 0:
                # might need to make weights absolute
                weights = np.abs(weights)

            results.loc[item] = np.dot(nbr_vs, n_rates) / np.sum(nbr_vs)

        results += model.means

        _logger.debug('user %s: predicted for %d of %d items',
                      user, results.notna().sum(), len(items))
        return results

    def save_model(self, model, file):
        with pd.HDFStore(file, 'w') as hdf:
            h5 = hdf._handle
            group = h5.create_group('/', 'ii-model')
            h5.create_array(group, 'items', model.items.values)
            h5.create_array(group, 'means', model.means.values)
            h5.create_array(group, 'col_ptrs', model.sim_matrix.indptr)
            h5.create_array(group, 'row_nums', model.sim_matrix.indices)
            h5.create_array(group, 'sim_values', model.sim_matrix.data)

            hdf['ratings'] = model.rating_matrix

    def load_model(self, file):
        with pd.HDFStore(file, 'r') as hdf:
            ratings = hdf['ratings']
            h5 = hdf._handle

            items = h5.get_node('/ii-model', 'items').read()
            items = pd.Index(items)
            means = h5.get_node('/ii-model', 'means').read()
            means = pd.Series(means, index=items)

            indptr = h5.get_node('/ii-model', 'col_ptrs').read()
            indices = h5.get_node('/ii-model', 'row_nums').read()
            values = h5.get_node('/ii-model', 'sim_values').read()

            matrix = sps.csr_matrix((values, indices, indptr))

            return IIModel(items, means, np.diff(indptr), matrix, ratings)
