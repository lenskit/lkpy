"""
Item-based k-NN collaborative filtering.
"""

import logging
import warnings

import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numba import njit, jitclass, prange, int32

from lenskit import util, matrix, DataWarning
from lenskit.util.accum import kvp_insert
from . import Predictor

_logger = logging.getLogger(__name__)


@jitclass({
    'counts': int32[:]
})
class _CountVisitor:
    def __init__(self, nrows):
        self.counts = np.zeros(nrows, dtype=np.int32)

    def visit(self, i, j, v):
        self.counts[i] += 1


@jitclass({
    'pointers': int32[:],
    'limits': int32[:],
    'out': matrix._CSR.class_type.instance_type
})
class _AccVisitor:
    def __init__(self, mat: matrix._CSR, nnbrs):
        self.out = mat
        self.limits = nnbrs
        self.pointers = mat.rowptrs[:-1].copy()

    def visit(self, i, j, v):
        sp = self.out.rowptrs[i]
        ep = self.pointers[i]
        ep = kvp_insert(sp, ep, self.limits[i], j, v, self.out.colinds, self.out.values)
        self.pointers[i] = ep


@njit(nogil=True)
def _visit_nbrs(smat: matrix._CSR, visitor, thresh: float, triangular: bool):
    "Count the number of neighbors passing the threshold for each row."
    for i in range(smat.nrows):
        sp, ep = smat.row_extent(i)
        for j in range(ep - sp):
            jp = sp + j
            c = smat.colinds[jp]
            v = smat.values[jp]
            if c != i and v >= thresh:
                visitor.visit(i, c, v)
                if triangular:
                    visitor.visit(c, i, v)


@njit(nogil=True)
def _predict_weighted_average(model, nitems, nrange, ratings, targets):
    "Weighted average prediction function"
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
    "Sum-of-similarities prediction function"
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
        sp_rmat = rmat.to_scipy()

        _logger.info('[%s] multiplying matrix with scipy', self._timer)
        smat = sp_rmat.T @ sp_rmat
        smat = matrix.CSR.from_scipy(smat, False)

        csr = self._filter_select(smat, False)
        return csr

    def _mkl_similarities(self, mkl, rmat):
        assert rmat.values is not None

        _logger.info('[%s] multiplying matrix with MKL', self._timer)
        smat = mkl.csr_syrk(rmat)

        smat = self._filter_select(smat, True)

        return smat

    def _filter_select(self, smat, triangular):
        "Threshold, filter, and symmetrify matrices"
        nitems = smat.nrows
        assert smat.ncols == nitems

        # Count possible neighbors
        cv = _CountVisitor(nitems)
        _visit_nbrs(smat.N, cv, self.min_sim, triangular)
        possible = cv.counts

        # Count neighbors to use neighbors
        min_nbrs = self.min_nbrs
        if min_nbrs is not None and min_nbrs > 0:
            nnbrs = np.maximum(possible, min_nbrs, dtype=np.int32)
        else:
            nnbrs = possible
        nsims = np.sum(nnbrs)
        _logger.info('truncating %d neighbors to %d', smat.nnz, nsims)

        # set up the target matrix
        trimmed = matrix.CSR.empty((nitems, nitems), nnbrs)

        # copy values into target arrays
        av = _AccVisitor(trimmed.N, nnbrs)
        _visit_nbrs(smat.N, av, self.min_sim, triangular)
        assert np.all(av.pointers == trimmed.rowptrs[1:])

        _logger.info('sorting neighborhoods')
        trimmed.sort_values()

        # and construct the new matrix
        return trimmed

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
