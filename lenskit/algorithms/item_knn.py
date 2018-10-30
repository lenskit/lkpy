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
        if self.save_neighbors is None:
            self.save_neighbors = -1

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
        self._timer = util.Stopwatch()

        rmat, users, items = matrix.sparse_ratings(ratings, scipy=True)
        n_items = len(items)
        _logger.info('[%s] made sparse matrix for %d items (%d ratings)',
                     self._timer, len(items), rmat.nnz)

        rmat, item_means = self._mean_center(ratings, rmat, items)

        rmat = self._normalize(rmat)

        _logger.info('[%s] computing similarity matrix', self._timer)
        rmat = matrix.csr_from_scipy(rmat, copy=False)
        smat = self._compute_similarities(rmat)

        _logger.info('[%s] got neighborhoods for %d of %d items',
                     self._timer, np.sum(np.diff(smat.rowptrs) > 0), n_items)

        _logger.info('[%s] computed %d neighbor pairs', self._timer, smat.nnz)

        return IIModel(items, item_means, np.diff(smat.rowptrs),
                       smat, ratings.set_index(['user', 'item']).rating)

    def _mean_center(self, ratings, rmat, items):
        item_means = ratings.groupby('item').rating.mean()
        item_means = item_means.reindex(items).values
        rmat.data = rmat.data - item_means[rmat.indices]
        _logger.info('[%s] computed means for %d items', self._timer, len(item_means))
        return rmat, item_means

    def _normalize(self, rmat):
        # compute column norms
        norms = spla.norm(rmat, 2, axis=0)
        # and multiply by a diagonal to normalize columns
        recip_norms = norms.copy()
        is_nz = recip_norms > 0
        recip_norms[is_nz] = np.reciprocal(recip_norms[is_nz])
        norm_mat = rmat @ sps.diags(recip_norms)
        assert norm_mat.shape[1] == rmat.shape[1]
        # and reset NaN
        norm_mat.data[np.isnan(norm_mat.data)] = 0
        _logger.info('[%s] normalized user-item ratings', self._timer)
        return norm_mat

    def _compute_similarities(self, rmat):
        mkl = matrix.mkl_ops()
        if mkl is None:
            return self._scipy_similarities(rmat)
        else:
            return self._mkl_similarities(mkl, rmat)

    def _scipy_similarities(self, rmat):
        nitems = rmat.ncols
        sp_rmat = matrix.csr_to_scipy(rmat)

        _logger.info('[%s] multiplying matrix', self._timer)
        smat = sp_rmat.T @ sp_rmat
        smat = smat.tocoo()
        rows, cols, vals = smat.row, smat.col, smat.data
        rows = rows[:smat.nnz]
        cols = cols[:smat.nnz]
        vals = vals[:smat.nnz]

        rows, cols, vals = self._filter_similarities(rows, cols, vals)
        csr = self._select_similarities(nitems, rows, cols, vals)
        return csr

    def _mkl_similarities(self, mkl, rmat):
        nitems = rmat.ncols

        _logger.info('[%s] multiplying matrix', self._timer)
        smat = mkl.csr_syrk(rmat)
        rows = matrix.csr_rowinds(smat)
        cols = smat.colinds
        vals = smat.values

        rows, cols, vals = self._filter_similarities(rows, cols, vals)
        nnz = len(rows)

        _logger.info('[%s] making matrix symmetric (%d nnz)', self._timer, nnz)
        rows.resize(nnz * 2)
        cols.resize(nnz * 2)
        vals.resize(nnz * 2)
        rows[nnz:] = cols[:nnz]
        cols[nnz:] = rows[:nnz]
        vals[nnz:] = vals[:nnz]

        csr = self._select_similarities(nitems, rows, cols, vals)
        return csr

    def _filter_similarities(self, rows, cols, vals):
        "Threshold similarites & remove self-similarities."
        _logger.info('[%s] filtering similarities', self._timer)
        # remove self-similarity
        mask = rows != cols

        # remove too-small similarities
        if self.min_similarity is not None:
            mask = mask & (vals >= self.min_similarity)

        _logger.info('[%s] filter keeps %d of %d entries', self._timer, np.sum(mask), len(rows))

        return rows[mask], cols[mask], vals[mask]

    def _select_similarities(self, nitems, rows, cols, vals):
        _logger.info('[%s] ordering similarities', self._timer)
        csr = matrix.csr_from_coo(rows, cols, vals, shape=(nitems, nitems))
        csr.sort_values()

        if self.save_neighbors is None or self.save_neighbors <= 0:
            return csr

        _logger.info('[%s] picking top similarities', self._timer)
        counts = csr.row_nnzs()
        ncounts = np.fmin(counts, self.save_neighbors)
        nnz = np.sum(ncounts)

        rp2 = np.zeros_like(csr.rowptrs)
        rp2[1:] = np.cumsum(ncounts)
        ci2 = np.zeros(nnz, np.int32)
        vs2 = np.zeros(nnz)
        for i in range(nitems):
            sp1 = csr.rowptrs[i]
            sp2 = rp2[i]

            ep1 = sp1 + counts[i]
            ep2 = sp2 + counts[i]
            ci2[sp2:ep2] = csr.colinds[sp1:ep1]
            vs2[sp2:ep2] = csr.values[sp1:ep1]

        return matrix.CSR(csr.nrows, csr.ncols, nnz, rp2, ci2, vs2)

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

        mat = model.sim_matrix
        row = matrix.csr_rowinds(mat)
        coo_df = pd.DataFrame({'item': row, 'neighbor': mat.colinds, 'similarity': mat.values})
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
        csr = matrix.csr_from_coo(coo_df['item'].values, coo_df['neighbor'].values,
                                  coo_df['similarity'].values,
                                  shape=(nitems, nitems))
        csr.sort_values()

        rmat = pd.read_parquet(str(path / 'ratings.parquet'))
        rmat = rmat.set_index(['user', 'item'])

        return IIModel(items, means, np.diff(csr.rowptrs), csr, rmat)

    def __str__(self):
        return 'ItemItem(nnbrs={}, msize={})'.format(self.max_neighbors, self.save_neighbors)
