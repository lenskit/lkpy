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
from numba import njit

from lenskit import util, matrix
from . import Trainable, Predictor

_logger = logging.getLogger(__name__)

IIModel = namedtuple('IIModel', ['items', 'means', 'counts', 'sim_matrix', 'rating_matrix'])


def _sort_and_truncate(nitems, smat, min_sim, nnbrs):
    assert smat.nrows == nitems
    assert smat.ncols == nitems

    ip2 = np.zeros(nitems + 1, dtype=np.int32)
    ind2 = smat.colinds
    d2 = smat.values

    _logger.debug('full matrix has %d entries', smat.nnz)

    # compute a first pass at the NNZ based on threshold
    if min_sim is not None:
        nnz = np.sum(smat.values[:smat.nnz] >= min_sim)
        ind2 = np.zeros(nnz, dtype=np.int32)
        d2 = np.zeros(nnz)

    for i in range(nitems):
        sp = smat.rowptrs[i]
        ep = smat.rowptrs[i+1]
        cols = smat.colinds[sp:ep]
        vals = smat.values[sp:ep]
        used = np.argsort(-vals)
        used = used[cols[used] != i]

        # filter
        if min_sim is not None:
            used = used[vals[used] >= min_sim]

        # truncate
        if nnbrs and nnbrs > 0:
            used = used[:nnbrs]

        sp = ip2[i]
        ep = sp + len(used)
        ip2[i+1] = ep
        ind2[sp:ep] = cols[used]
        d2[sp:ep] = vals[used]

    nnz = ip2[-1]
    _logger.debug('truncating to %d entries', nnz)
    ind2.resize(nnz)
    d2.resize(nnz)
    return matrix.CSR(nitems, nitems, nnz, ip2, ind2, d2)


@njit(nogil=True)
def _predict(model, nitems, nrange, ratings, targets):
    min_nbrs, max_nbrs = nrange
    scores = np.full(nitems, np.nan, dtype=np.float_)

    for i in range(targets.shape[0]):
        iidx = targets[i]
        rptr = model.rowptrs[iidx]
        rend = model.rowptrs[iidx + 1]

        num = 0
        denom = 0
        nnbrs = 0

        for j in range(rptr, rend):
            nidx = model.colinds[j]
            if np.isnan(ratings[nidx]):
                continue

            nnbrs = nnbrs + 1
            num = num + ratings[nidx] * model.values[j]
            denom = denom + np.abs(model.values[j])

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
        assert rmat.shape[0] == len(users)
        assert rmat.shape[1] == n_items

        # compute column norms
        norms = spla.norm(rmat, 2, axis=0)
        # and multiply by a diagonal to normalize columns
        recip_norms = norms.copy()
        is_nz = recip_norms > 0
        recip_norms[is_nz] = np.reciprocal(recip_norms[is_nz])
        norm_mat = rmat @ sps.diags(recip_norms)
        assert norm_mat.shape[1] == n_items
        # and reset NaN
        norm_mat.data[np.isnan(norm_mat.data)] = 0
        _logger.info('[%s] normalized user-item ratings', watch)
        _logger.info('[%s] computing similarity matrix', watch)
        smat = matrix.csr_syrk(norm_mat)
        _logger.info('[%s] truncating similarity matrix', watch)
        smat = _sort_and_truncate(n_items, smat, self.min_similarity, self.save_neighbors)

        _logger.info('[%s] got neighborhoods for %d of %d items',
                     watch, np.sum(np.diff(smat.rowptrs) > 0), n_items)

        _logger.info('[%s] computed %d neighbor pairs', watch, smat.nnz)

        return IIModel(items, item_means, np.diff(smat.rowptrs),
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
        iscore = _predict(model.sim_matrix,
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

        coo = matrix.csr_to_scipy(model.sim_matrix).tocoo()
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
        csr = matrix.csr_from_coo(coo_df['item'].values, coo_df['neighbor'].values, coo_df['similarity'].values,
                                  shape=(nitems, nitems))

        for i in range(nitems):
            sp = csr.rowptrs[i]
            ep = csr.rowptrs[i+1]
            if ep == sp:
                continue

            ord = np.argsort(csr.values[sp:ep])
            ord = ord[::-1]
            csr.colinds[sp:ep] = csr.colinds[sp + ord]
            csr.values[sp:ep] = csr.values[sp + ord]

        rmat = pd.read_parquet(str(path / 'ratings.parquet'))
        rmat = rmat.set_index(['user', 'item'])

        return IIModel(items, means, np.diff(csr.rowptrs), csr, rmat)

    def __str__(self):
        return 'ItemItem(nnbrs={}, msize={})'.format(self.max_neighbors, self.save_neighbors)
