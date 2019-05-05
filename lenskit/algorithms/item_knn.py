"""
Item-based k-NN collaborative filtering.
"""

import pathlib
import logging
import warnings

import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numba import njit, prange

from lenskit import util, matrix, DataWarning
from . import Predictor

_logger = logging.getLogger(__name__)


@njit(nogil=True)
def _predict_weighted_average(model, nitems, nrange, ratings, targets):
    min_nbrs, max_nbrs = nrange
    scores = np.full(nitems, np.nan, dtype=np.float_)

    for i in prange(targets.shape[0]):
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


@njit(nogil=True)
def _predict_sum(model, nitems, nrange, ratings, targets):
    min_nbrs, max_nbrs = nrange
    scores = np.full(nitems, np.nan, dtype=np.float_)

    for i in prange(targets.shape[0]):
        iidx = targets[i]
        rptr = model.rowptrs[iidx]
        rend = model.rowptrs[iidx + 1]

        score = 0
        nnbrs = 0

        for j in range(rptr, rend):
            nidx = model.colinds[j]
            if np.isnan(ratings[nidx]):
                continue

            nnbrs = nnbrs + 1
            score = score + model.values[j]

            if max_nbrs > 0 and nnbrs >= max_nbrs:
                break

        if nnbrs < min_nbrs:
            continue

        scores[iidx] = score

    return scores


_predictors = {
    'weighted-average': _predict_weighted_average,
    'sum': _predict_sum
}


class ItemItem(Predictor):
    """
    Item-item nearest-neighbor collaborative filtering with ratings. This item-item implementation
    is not terribly configurable; it hard-codes design decisions found to work well in the previous
    Java-based LensKit code.

    Attributes:
        item_index_(pandas.Index): the index of item IDs.
        item_means_(numpy.ndarray): the mean rating for each known item.
        item_counts_(numpy.ndarray): the number of saved neighbors for each item.
        sim_matrix_(matrix.CSR): the similarity matrix.
        user_index_(pandas.Index): the index of known user IDs for the rating matrix.
        rating_matrix_(matrix.CSR): the user-item rating matrix for looking up users' ratings.
    """

    def __init__(self, nnbrs, min_nbrs=1, min_sim=1.0e-6, save_nbrs=None,
                 center=True, aggregate='weighted-average'):
        """
        Args:
            nnbrs(int):
                the maximum number of neighbors for scoring each item (``None`` for unlimited)
            min_nbrs(int): the minimum number of neighbors for scoring each item
            min_sim(double): minimum similarity threshold for considering a neighbor
            save_nbrs(double):
                the number of neighbors to save per item in the trained model
                (``None`` for unlimited)
            center(bool):
                whether to normalize (mean-center) rating vectors.  Turn this off when working
                with unary data and other data types that don't respond well to centering.
            aggregate:
                the type of aggregation to do. Can be ``weighted-average`` or ``sum``.
        """
        self.nnbrs = nnbrs
        if self.nnbrs is not None and self.nnbrs < 1:
            self.nnbrs = -1
        self.min_nbrs = min_nbrs
        if self.min_nbrs is not None and self.min_nbrs < 1:
            self.min_nbrs = 1
        self.min_sim = min_sim
        self.save_nbrs = save_nbrs
        self.center = center
        self.aggregate = aggregate
        try:
            self._predict_agg = _predictors[aggregate]
        except KeyError:
            raise ValueError('unknown aggregator {}'.format(aggregate))

    def fit(self, ratings):
        """
        Train a model.

        The model-training process depends on ``save_nbrs`` and ``min_sim``, but *not* on other
        algorithm parameters.

        Args:
            ratings(pandas.DataFrame):
                (user,item,rating) data for computing item similarities.
        """
        # Training proceeds in 2 steps:
        # 1. Normalize item vectors to be mean-centered and unit-normalized
        # 2. Compute similarities with pairwise dot products
        self._timer = util.Stopwatch()

        init_rmat, users, items = matrix.sparse_ratings(ratings)
        n_items = len(items)
        _logger.info('[%s] made sparse matrix for %d items (%d ratings from %d users)',
                     self._timer, len(items), init_rmat.nnz, len(users))

        rmat, item_means = self._mean_center(ratings, init_rmat, items)

        rmat = self._normalize(rmat)

        _logger.info('[%s] computing similarity matrix', self._timer)
        smat = self._compute_similarities(rmat)

        _logger.info('[%s] got neighborhoods for %d of %d items',
                     self._timer, np.sum(np.diff(smat.rowptrs) > 0), n_items)

        _logger.info('[%s] computed %d neighbor pairs', self._timer, smat.nnz)

        self.item_index_ = items
        self.item_means_ = item_means
        self.item_counts_ = np.diff(smat.rowptrs)
        self.sim_matrix_ = smat
        self.user_index_ = users
        self.rating_matrix_ = init_rmat

        return self

    def _mean_center(self, ratings, rmat, items):
        if not self.center:
            return rmat, None

        item_means = ratings.groupby('item').rating.mean()
        item_means = item_means.reindex(items).values
        mcvals = rmat.values - item_means[rmat.colinds]
        nmat = matrix.CSR(rmat.nrows, rmat.ncols, rmat.nnz,
                          rmat.rowptrs.copy(), rmat.colinds.copy(), mcvals)
        _logger.info('[%s] computed means for %d items', self._timer, len(item_means))
        return nmat, item_means

    def _normalize(self, rmat):
        rmat = rmat.to_scipy()
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
        _logger.info('[%s] normalized rating matrix columns', self._timer)
        return matrix.CSR.from_scipy(norm_mat, False)

    def _compute_similarities(self, rmat):
        mkl = matrix.mkl_ops()
        if mkl is None:
            return self._scipy_similarities(rmat)
        else:
            return self._mkl_similarities(mkl, rmat)

    def _scipy_similarities(self, rmat):
        nitems = rmat.ncols
        sp_rmat = rmat.to_scipy()

        _logger.info('[%s] multiplying matrix with scipy', self._timer)
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
        assert rmat.values is not None

        _logger.info('[%s] multiplying matrix with MKL', self._timer)
        smat = mkl.csr_syrk(rmat)
        rows = smat.rowinds()
        cols = smat.colinds
        vals = smat.values

        rows, cols, vals = self._filter_similarities(rows, cols, vals)
        del smat
        nnz = len(rows)

        _logger.info('[%s] making matrix symmetric (%d -> %d nnz)',
                     self._timer, nnz, nnz * 2)
        rows = np.resize(rows, nnz * 2)
        cols = np.resize(cols, nnz * 2)
        vals = np.resize(vals, nnz * 2)
        rows[nnz:] = cols[:nnz]
        cols[nnz:] = rows[:nnz]
        vals[nnz:] = vals[:nnz]

        csr = self._select_similarities(nitems, rows, cols, vals)
        return csr

    def _filter_similarities(self, rows, cols, vals):
        "Threshold similarites & remove self-similarities."
        _logger.info('[%s] filtering %d similarities', self._timer, len(rows))
        # remove self-similarity
        mask = rows != cols

        # remove too-small similarities
        if self.min_sim is not None:
            mask = np.logical_and(mask, vals >= self.min_sim)

        _logger.info('[%s] filter keeps %d of %d entries', self._timer, np.sum(mask), len(rows))

        return rows[mask], cols[mask], vals[mask]

    def _select_similarities(self, nitems, rows, cols, vals):
        _logger.info('[%s] ordering similarities', self._timer)
        csr = matrix.CSR.from_coo(rows, cols, vals, shape=(nitems, nitems))
        csr.sort_values()

        if self.save_nbrs is None or self.save_nbrs <= 0:
            return csr

        _logger.info('[%s] picking %d top similarities', self._timer, self.save_nbrs)
        counts = csr.row_nnzs()
        _logger.debug('have %d rows in size range [%d,%d]',
                      len(counts), np.min(counts), np.max(counts))
        ncounts = np.fmin(counts, self.save_nbrs)
        _logger.debug('will have %d rows in size range [%d,%d]',
                      len(ncounts), np.min(ncounts), np.max(ncounts))
        assert np.all(ncounts <= self.save_nbrs)
        assert np.all(ncounts >= 0)
        nnz = np.sum(ncounts)

        rp2 = np.zeros_like(csr.rowptrs)
        rp2[1:] = np.cumsum(ncounts)
        ci2 = np.zeros(nnz, np.int32)
        vs2 = np.zeros(nnz)
        for i in range(nitems):
            sp1 = csr.rowptrs[i]
            sp2 = rp2[i]

            ep1 = sp1 + ncounts[i]
            ep2 = sp2 + ncounts[i]
            assert ep1 - sp1 == ep2 - sp2

            ci2[sp2:ep2] = csr.colinds[sp1:ep1]
            vs2[sp2:ep2] = csr.values[sp1:ep1]

        return matrix.CSR(csr.nrows, csr.ncols, nnz, rp2, ci2, vs2)

    def predict_for_user(self, user, items, ratings=None):
        _logger.debug('predicting %d items for user %s', len(items), user)
        if ratings is None:
            if user not in self.user_index_:
                _logger.debug('user %s missing, returning empty predictions', user)
                return pd.Series(np.nan, index=items)
            upos = self.user_index_.get_loc(user)
            ratings = pd.Series(self.rating_matrix_.row_vs(upos),
                                index=pd.Index(self.item_index_[self.rating_matrix_.row_cs(upos)]))

        if not ratings.index.is_unique:
            wmsg = 'user {} has duplicate ratings, this is likely to cause problems'.format(user)
            warnings.warn(wmsg, DataWarning)

        # set up rating array
        # get rated item positions & limit to in-model items
        ri_pos = self.item_index_.get_indexer(ratings.index)
        m_rates = ratings[ri_pos >= 0]
        ri_pos = ri_pos[ri_pos >= 0]
        rate_v = np.full(len(self.item_index_), np.nan, dtype=np.float_)
        # mean-center the rating array
        if self.center:
            rate_v[ri_pos] = m_rates.values - self.item_means_[ri_pos]
        else:
            rate_v[ri_pos] = m_rates.values
        _logger.debug('user %s: %d of %d rated items in model', user, len(ri_pos), len(ratings))
        assert np.sum(np.logical_not(np.isnan(rate_v))) == len(ri_pos)

        # set up item result vector
        # ipos will be an array of item indices
        i_pos = self.item_index_.get_indexer(items)
        i_pos = i_pos[i_pos >= 0]
        _logger.debug('user %s: %d of %d requested items in model', user, len(i_pos), len(items))

        # scratch result array
        iscore = np.full(len(self.item_index_), np.nan, dtype=np.float_)

        # now compute the predictions
        iscore = self._predict_agg(self.sim_matrix_.N,
                                   len(self.item_index_),
                                   (self.min_nbrs, self.nnbrs),
                                   rate_v, i_pos)

        nscored = np.sum(np.logical_not(np.isnan(iscore)))
        if self.center:
            iscore += self.item_means_
        assert np.sum(np.logical_not(np.isnan(iscore))) == nscored

        results = pd.Series(iscore, index=self.item_index_)
        results = results[results.notna()]
        results = results.reindex(items, fill_value=np.nan)
        assert results.notna().sum() == nscored

        _logger.debug('user %s: predicted for %d of %d items',
                      user, results.notna().sum(), len(items))

        return results

    def __str__(self):
        return 'ItemItem(nnbrs={}, msize={})'.format(self.nnbrs, self.save_nbrs)
