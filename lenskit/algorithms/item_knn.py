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
import numba as n
from numba import njit, jitclass

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
def __train_row(context, thresh, nnbrs, item):
    work = np.zeros(context.n_items)
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
    idx, = np.where(mask)
    if nnbrs > 0:
        acc = util.Accumulator(work, nnbrs)
        acc.add_all(idx)
        top = acc.top_keys()
        return (top, work[top].copy())
    else:
        sims = work[idx]
        order = sims.argsort()
        order = order[::-1]
        return (idx[order].astype(np.int32), sims[order])


@njit
def _train(context, thresh, nnbrs):
    nrows = []
    srows = []

    for item in range(context.n_items):
        nrow, srow = __train_row(context, thresh, nnbrs, item)

        nrows.append(nrow)
        srows.append(srow)

    counts = np.array([len(n) for n in nrows], dtype=np.int32)
    cells = np.sum(counts)

    # assemble our results in to a CSR
    ptrs = np.zeros(len(nrows) + 1, dtype=np.int32)
    ptrs[1:] = np.cumsum(counts)
    assert ptrs[context.n_items] == cells

    indices = np.empty(cells, dtype=np.int32)
    sims = np.empty(cells)
    for i in range(context.n_items):
        sp = ptrs[i]
        ep = ptrs[i+1]
        assert counts[i] == ep - sp
        indices[sp:ep] = nrows[i]
        sims[sp:ep] = srows[i]

    return (ptrs, indices, sims)


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
        n_items = len(items)
        item_means = item_means.reindex(items).values
        _logger.info('[%s] made sparse matrix for %d items (%d ratings)',
                     watch, len(items), rmat.nnz)

        # stupid trick: indices are items, look up means, subtract!
        rmat.data = rmat.data - item_means[rmat.indices]

        # compute column norms
        norms = spla.norm(rmat, 2, axis=0)
        # and multiply by a diagonal to normalize columns
        norm_mat = rmat @ sps.diags(np.reciprocal(norms))
        # and reset NaN
        norm_mat.data[np.isnan(norm_mat.data)] = 0
        _logger.info('[%s] normalized user-item ratings', watch)

        _logger.info('[%s] computing similarity matrix', watch)
        ptr, nbr, sim = _train(_make_context(norm_mat),
                               self.min_similarity,
                               self.save_neighbors
                               if self.save_neighbors and self.save_neighbors > 0
                               else -1)

        _logger.info('[%s] got neighborhoods for %d of %d items',
                     watch, np.sum(np.diff(ptr) > 0), n_items)
        smat = sps.csr_matrix((sim, nbr, ptr), shape=(n_items, n_items))

        _logger.info('[%s] computed %d neighbor pairs', watch, smat.nnz)

        return IIModel(items, item_means, np.diff(smat.indptr),
                       smat, ratings.set_index(['user', 'item']).rating)

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
