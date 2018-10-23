"""
Item-based k-NN collaborative filtering.
"""

import pathlib
from collections import namedtuple
import logging

import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numba import njit, prange, objmode

from lenskit import util, matrix
from . import Trainable, Predictor

_logger = logging.getLogger(__name__)

IIModel = namedtuple('IIModel', ['items', 'means', 'counts', 'sim_matrix', 'rating_matrix'])
IIModel._matrix = property(lambda x: (x.sim_matrix.indptr, x.sim_matrix.indices, x.sim_matrix.data))


@njit(nogil=True)
def __train_row(rmat: matrix.CSR, item_users: matrix.CSR, thresh, nnbrs, item):
    work = np.zeros(rmat.ncols)
    iu_rp = rmat.rowptrs
    # iterate the users who have rated this item
    for uidx in range(item_users.rowptrs[item], item_users.rowptrs[item+1]):
        u = item_users.colinds[uidx]
        # find user's rating for this item
        urp = -1
        for iidx in range(iu_rp[u], iu_rp[u+1]):
            if rmat.colinds[iidx] == item:
                urp = iidx
                ur = rmat.values[urp]
                break

        # accumulate pieces of dot products
        for iidx in range(rmat.rowptrs[u], rmat.rowptrs[u+1]):
            nbr = rmat.colinds[iidx]
            if nbr != item:
                work[nbr] = work[nbr] + ur * rmat.values[iidx]

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


@njit(nogil=True, parallel=True)
def __build_matrix(rmat, thresh, nnbrs):
    nitems = rmat.ncols
    _n_ph = np.array([0], dtype=np.int32)
    _s_ph = np.array([1.0], dtype=np.float_)
    nrows = [_n_ph for _ in range(nitems)]
    srows = [_s_ph for _ in range(nitems)]
    item_users = rmat.transpose_coords()

    nbatches = nitems // 100

    for batch in prange(nbatches):
        bs = batch * 100
        be = bs + 100
        if be > nitems:
            be = nitems

        for item in range(bs, be):
            nrow, srow = __train_row(rmat, item_users, thresh, nnbrs, item)

            nrows[item] = nrow
            srows[item] = srow

    return (nrows, srows)


@njit
def _train(rmat: matrix.CSR, thresh: float, nnbrs: int):
    nitems = rmat.ncols

    with objmode():
        _logger.info('starting parallel similarity build')

    nrows, srows = __build_matrix(rmat, thresh, nnbrs)

    with objmode():
        _logger.info('processing similarity results')

    counts = np.array([len(n) for n in nrows], dtype=np.int32)
    cells = np.sum(counts)

    # assemble our results in to a CSR
    ptrs = np.zeros(nitems + 1, dtype=np.int32)
    ptrs[1:] = np.cumsum(counts)

    with objmode():
        _logger.info('assembling sparse matrix with %d entries for %d rows', cells, nitems)

    assert ptrs[nitems] == cells

    indices = np.empty(cells, dtype=np.int32)
    sims = np.empty(cells)
    for i in range(nitems):
        sp = ptrs[i]
        ep = ptrs[i+1]
        assert counts[i] == ep - sp
        indices[sp:ep] = nrows[i]
        sims[sp:ep] = srows[i]

    return (ptrs, indices, sims)


@njit(nogil=True)
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

        rmat, users, items = matrix.sparse_ratings(ratings, scipy=True)
        n_items = len(items)
        item_means = item_means.reindex(items).values
        _logger.info('[%s] made sparse matrix for %d items (%d ratings)',
                     watch, len(items), rmat.nnz)

        # stupid trick: indices are items, look up means, subtract!
        rmat.data = rmat.data - item_means[rmat.indices]

        # compute column norms
        norms = spla.norm(rmat, 2, axis=0)
        # and multiply by a diagonal to normalize columns
        recip_norms = norms.copy()
        is_nz = recip_norms > 0
        recip_norms[is_nz] = np.reciprocal(recip_norms[is_nz])
        norm_mat = rmat @ sps.diags(recip_norms)
        # and reset NaN
        norm_mat.data[np.isnan(norm_mat.data)] = 0
        _logger.info('[%s] normalized user-item ratings', watch)
        rmat = matrix.csr_from_scipy(norm_mat)
        _logger.info('[%s] computing similarity matrix', watch)
        ptr, nbr, sim = _train(rmat, self.min_similarity,
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

    def save_model(self, model, path):
        path = pathlib.Path(path)
        _logger.info('saving I-I model to %s', path)
        path.mkdir(parents=True, exist_ok=True)

        imeans = pd.DataFrame({'item': model.items.values, 'mean': model.means})
        imeans.to_parquet(str(path / 'items.parquet'))

        coo = model.sim_matrix.tocoo()
        coo_df = pd.DataFrame({'item': coo.row, 'neighbor': coo.col, 'similarity': coo.data})
        coo_df.to_parquet(str(path / 'similarities.parquet'))

        model.rating_matrix.reset_index().to_parquet(str(path / 'ratings.parquet'))

    def load_model(self, path):
        path = pathlib.Path(path)
        _logger.info('loading I-I model from %s', path)

        imeans = pd.read_parquet(str(path / 'items.parquet'))
        items = pd.Index(imeans.item)
        means = imeans['mean'].values
        nitems = len(items)

        coo_df = pd.read_parquet(str(path / 'similarities.parquet'))
        _logger.info('read %d similarities for %d items', len(coo_df), nitems)
        csr = sps.csr_matrix((coo_df['similarity'].values,
                              (coo_df['item'].values, coo_df['neighbor'].values)),
                             (nitems, nitems))

        for i in range(nitems):
            sp = csr.indptr[i]
            ep = csr.indptr[i+1]
            if ep == sp:
                continue

            ord = np.argsort(csr.data[sp:ep])
            ord = ord[::-1]
            csr.indices[sp:ep] = csr.indices[sp + ord]
            csr.data[sp:ep] = csr.data[sp + ord]

        rmat = pd.read_parquet(str(path / 'ratings.parquet'))
        rmat = rmat.set_index(['user', 'item'])

        return IIModel(items, means, np.diff(csr.indptr), csr, rmat)

    def __str__(self):
        return 'ItemItem(nnbrs={}, msize={})'.format(self.max_neighbors, self.save_neighbors)
