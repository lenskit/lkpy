"""
Item-based k-NN collaborative filtering.
"""

from collections import namedtuple
import logging

import ctypes
import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numba import njit

from lenskit import util, matrix
from . import Trainable, Predictor

_logger = logging.getLogger(__package__)

IIModel = namedtuple('IIModel', ['items', 'means', 'counts', 'sim_matrix', 'rating_matrix'])
IIModel._matrix = property(lambda x: (x.sim_matrix.indptr, x.sim_matrix.indices, x.sim_matrix.data))

_IIContext = namedtuple('_IIContext', [
    'uptrs', 'items', 'ratings',
    'r_iptrs', 'r_users',
    'n_users', 'n_items'
])


def _make_context(matrix):
    csc = matrix.tocsc(copy=False)
    assert sps.isspmatrix_csc(csc)
    return _IIContext(matrix.indptr, matrix.indices, matrix.data,
                      csc.indptr, csc.indices,
                      matrix.shape[0], matrix.shape[1])


@njit
def _train(context, thresh, nnbrs):
    neighborhoods = []
    idx = np.arange(np.int32(context.n_items))
    work = np.zeros(context.n_items)

    for item in range(context.n_items):
        work.fill(0)
        for uidx in range(context.r_iptrs[item], context.r_iptrs[item+1]):
            u = context.r_users[uidx]
            # find user's rating for this item
            urp = -1
            for iidx in range(context.uptrs[u], context.uptrs[u+1]):
                if context.items[iidx] == item:
                    urp = iidx
                    ur = context.ratings[urp]
                    break
            assert urp >= 0

            # accumulate pieces of dot products
            for iidx in range(context.uptrs[u], context.uptrs[u+1]):
                nbr = context.items[iidx]
                if nbr != item:
                    work[nbr] = work[nbr] + ur * context.ratings[iidx]

        # now copy the accepted values into the results
        mask = work >= thresh
        if nnbrs > 0:
            acc = util.Accumulator(work, nnbrs)
            acc.add_all(idx[mask])
            top = acc.top_keys()
            neighborhoods.append((top, work[top]))
        else:
            neighborhoods.append((idx[mask], work[mask]))

    return neighborhoods


@njit
def _predict(model, nitems, nrange, ratings, targets):
    indptr, indices, similarity = model
    min_nbrs, max_nbrs = nrange
    scores = np.full(nitems, np.nan, dtype=np.float_)

    for i in range(targets.shape[0]):
        iidx = targets[i]
        rptr = indptr[iidx]
        rend = indptr[iidx + 1]

        num = 0
        denom = 0
        nnbrs = 0

        for j in range(rptr, rend):
            nidx = indices[j]
            if np.isnan(ratings[nidx]):
                continue

            nnbrs = nnbrs + 1
            num = num + ratings[nidx] * similarity[j]
            denom = denom + np.abs(similarity[j])

            if max_nbrs > 0 and nnbrs >= max_nbrs:
                break

        if nnbrs < min_nbrs:
            continue

        scores[iidx] = num / denom

    return scores


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

        rmat, users, items = matrix.sparse_ratings(ratings)
        item_means = item_means.reindex(items).values
        _logger.info('[%s] made sparse matrix for %d items (%d ratings)',
                     watch, len(items), rmat.nnz)

        # stupid trick: indices are items, look up means, subtract!
        rmat.data = rmat.data - item_means[rmat.indices]
        m2 = rmat.mean(axis=0)
        _logger.info('min mean: %f, max mean: %f', m2.A1.min(), m2.A1.max())

        # compute column norms
        norms = spla.norm(rmat, 2, axis=0)
        # and multiply by a diagonal to normalize columns
        norm_mat = rmat @ sps.diags(np.reciprocal(norms))
        # and reset NaN
        norm_mat.data[np.isnan(norm_mat.data)] = 0
        _logger.info('[%s] normalized user-item ratings', watch)

        _logger.info('[%s] computing similarity matrix', watch)
        sim_matrix = self._cy_matrix(norm_mat, watch)

        _logger.info('[%s] computed %d neighbor pairs', watch, sim_matrix.nnz)

        return IIModel(items, item_means, np.diff(sim_matrix.indptr),
                       sim_matrix, ratings.set_index(['user', 'item']).rating)

    def _cy_matrix(self, rmat, watch):
        _logger.debug('[%s] preparing Cython data launch', watch)
        # the Cython implementation requires contiguous numeric IDs.
        # so let's make those

        n_items = rmat.shape[1]

        _logger.debug('[%s] running accelerated matrix computations', watch)
        nbrhoods = _train(_make_context(rmat),
                          self.min_similarity,
                          self.save_neighbors
                          if self.save_neighbors and self.save_neighbors > 0
                          else -1)
        ndf = pd.concat(pd.DataFrame({'item': i, 'neighbor': ns, 'similarity': ss})
                        for (i, (ns, ss)) in enumerate(nbrhoods))

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
        _logger.info('[%s] sorted neighborhoods', watch)

        return smat

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
        iscore = _predict(model._matrix,
                          len(model.items),
                          (self.min_neighbors, self.max_neighbors),
                          rate_v, i_pos)

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
