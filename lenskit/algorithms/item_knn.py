"""
Item-based k-NN collaborative filtering.
"""

from sys import intern
import logging
import warnings

import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numba import njit, prange, objmode

from lenskit import util, matrix, DataWarning
from lenskit.util.accum import kvp_minheap_insert, kvp_minheap_sort
from . import Predictor

_logger = logging.getLogger(__name__)
_mkl_ops = matrix.mkl_ops()

if _mkl_ops is not None:
    # we have to import LK CFFI utils into this module
    for lkv in dir(_mkl_ops):
        if lkv.startswith('_lk_mkl'):
            globals()[lkv] = getattr(_mkl_ops, lkv)


@njit
def _make_blocks(n, size):
    "Create blocks for the range 0..n."
    blk_sp = np.arange(0, n, size)
    blk_ep = blk_sp + size
    blk_ep[-1] = n
    return (blk_sp, blk_ep)


@njit(nogil=True)
def _count_nbrs(mat: matrix._CSR, thresh: float):
    "Count the number of neighbors passing the threshold for each row."
    counts = np.zeros(mat.nrows, dtype=np.int32)
    cs = mat.colinds
    vs = mat.values
    for i in range(mat.nrows):
        sp, ep = mat.row_extent(i)
        for j in range(sp, ep):
            c = cs[j]
            v = vs[j]
            if c != i and v >= thresh:
                counts[i] += 1

    return counts


@njit
def _insert(dst, used, limits, i, c, v):
    "Insert one item into a heap"
    sp = dst.rowptrs[i]
    ep = sp + used[i]
    ep = kvp_minheap_insert(sp, ep, limits[i], c, v, dst.colinds, dst.values)
    used[i] = ep - sp


@njit(nogil=True)
def _copy_nbrs(src: matrix._CSR, dst: matrix._CSR, limits, thresh: float):
    "Copy neighbors into the output matrix."
    used = np.zeros(dst.nrows, dtype=np.int32)

    for i in range(src.nrows):
        sp, ep = src.row_extent(i)

        for j in range(sp, ep):
            c = src.colinds[j]
            v = src.values[j]
            if c != i and v >= thresh:
                _insert(dst, used, limits, i, c, v)

    return used


@njit(nogil=True, parallel=True)
def _sort_nbrs(smat):
    for i in prange(smat.nrows):
        sp, ep = smat.row_extent(i)
        kvp_minheap_sort(sp, ep, smat.colinds, smat.values)


@njit
def _sim_block(inb, rmh, min_sim, max_nbrs, nitems):
    "Compute a single block of the similarity matrix"
    rmat, bsp, bep = inb
    # assert rmat.nrows == bep - bsp

    if rmat.nnz == 0:
        return matrix._empty_csr(rmat.nrows, nitems, np.zeros(rmat.nrows, np.int32))

    # create a matrix handle for the subset matrix
    amh = _mkl_ops._from_csr(rmat)
    _lk_mkl_spopt(amh)

    smh = _lk_mkl_spmabt(rmh, amh)
    _lk_mkl_spfree(amh)

    _lk_mkl_sporder(smh)  # for reproducibility

    block = _lk_mkl_spexport_p(smh)
    bnr = _lk_mkl_spe_nrows(block)
    bnc = _lk_mkl_spe_ncols(block)
    # bnr and bnc should be right
    # assert bnc == bep - bsp

    r_sp = _lk_mkl_spe_row_sp(block)
    r_ep = _lk_mkl_spe_row_ep(block)
    r_cs = _lk_mkl_spe_colinds(block)
    r_vs = _lk_mkl_spe_values(block)

    # pass 1: compute the size of each row
    sizes = np.zeros(rmat.nrows, np.int32)
    for i in range(bnr):
        for j in range(r_sp[i], r_ep[i]):
            # we accept the neighbor if it passes threshold and isn't a self-similarity
            r = r_cs[j]
            if i != bsp + r and r_vs[j] >= min_sim:
                sizes[r] += 1

    if max_nbrs > 0:
        for i in range(rmat.nrows):
            if sizes[i] > max_nbrs:
                sizes[i] = max_nbrs

    if bnc == 0:
        # empty resulting matrix, oops
        return matrix._empty_csr(rmat.nrows, nitems, np.zeros(rmat.nrows, np.int32))

    # allocate a matrix
    block_csr = matrix._empty_csr(bnc, bnr, sizes)

    # pass 2: truncate each row into the matrix
    eps = block_csr.rowptrs[:-1].copy()
    for c in range(bnr):
        for j in range(r_sp[c], r_ep[c]):
            v = r_vs[j]
            r = r_cs[j]
            sp, lep = block_csr.row_extent(r)
            lim = lep - sp
            if c != bsp + r and v >= min_sim:
                eps[r] = kvp_minheap_insert(sp, eps[r], lim, c, v,
                                            block_csr.colinds, block_csr.values)
        # we're done!
        # assert lim == ep - sp

    _lk_mkl_spe_free(block)
    _lk_mkl_spfree(smh)
    return block_csr


@njit(nogil=True, parallel=True)
def _mkl_sim_blocks(trmat, min_sim, max_nbrs):
    "Compute the similarity matrix with blocked MKL calls"
    nitems = trmat.nrows
    blk_sp, blk_ep = _make_blocks(nitems, 500)
    nblocks = len(blk_sp)
    with objmode():
        _logger.info('split %d items into %d blocks', nitems, nblocks)
        _logger.info('matrices have %d nnz', trmat.nnz)
    rmat_h = _mkl_ops._from_csr(trmat)
    _lk_mkl_sporder(rmat_h)
    _lk_mkl_spopt(rmat_h)

    blocks = [(trmat.subset_rows(blk_sp[bi], blk_ep[bi]), blk_sp[bi], blk_ep[bi])
              for bi in range(nblocks)]

    for bi in prange(nblocks):
        b, bs, be = blocks[bi]
        bres = _sim_block(blocks[bi], rmat_h, min_sim, max_nbrs, nitems)
        blocks[bi] = (bres, bs, be)

    _lk_mkl_spfree(rmat_h)

    return blocks  # we'll do the rest of the work in Python


@njit(nogil=True)
def _predict_weighted_average(model, nitems, nrange, ratings, rated, targets):
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
            if not rated[nidx]:
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
def _predict_sum(model, nitems, nrange, ratings, rated, targets):
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
            if not rated[nidx]:
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

    Attributes:
        item_index_(pandas.Index): the index of item IDs.
        item_means_(numpy.ndarray): the mean rating for each known item.
        item_counts_(numpy.ndarray): the number of saved neighbors for each item.
        sim_matrix_(matrix.CSR): the similarity matrix.
        user_index_(pandas.Index): the index of known user IDs for the rating matrix.
        rating_matrix_(matrix.CSR): the user-item rating matrix for looking up users' ratings.
    """
    AGG_SUM = intern('sum')
    AGG_WA = intern('weighted-average')
    _use_mkl = True

    def __init__(self, nnbrs, min_nbrs=1, min_sim=1.0e-6, save_nbrs=None,
                 center=True, aggregate='weighted-average'):
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

        _logger.debug('[%s] beginning fit, memory use %s', self._timer, util.max_memory())

        init_rmat, users, items = matrix.sparse_ratings(ratings)
        n_items = len(items)
        _logger.info('[%s] made sparse matrix for %d items (%d ratings from %d users)',
                     self._timer, len(items), init_rmat.nnz, len(users))
        _logger.debug('[%s] made matrix, memory use %s', self._timer, util.max_memory())

        rmat, item_means = self._mean_center(ratings, init_rmat, items)
        _logger.debug('[%s] centered, memory use %s', self._timer, util.max_memory())

        rmat = self._normalize(rmat)
        _logger.debug('[%s] normalized, memory use %s', self._timer, util.max_memory())

        _logger.info('[%s] computing similarity matrix', self._timer)
        smat = self._compute_similarities(rmat)
        _logger.debug('[%s] computed, memory use %s', self._timer, util.max_memory())

        _logger.info('[%s] got neighborhoods for %d of %d items',
                     self._timer, np.sum(np.diff(smat.rowptrs) > 0), n_items)

        _logger.info('[%s] computed %d neighbor pairs', self._timer, smat.nnz)

        self.item_index_ = items
        self.item_means_ = item_means
        self.item_counts_ = np.diff(smat.rowptrs)
        self.sim_matrix_ = smat
        self.user_index_ = users
        self.rating_matrix_ = init_rmat
        # create an inverted similarity matrix for efficient scanning
        self._sim_inv_ = smat.transpose()
        _logger.info('[%s] transposed matrix for optimization', self._timer)
        _logger.debug('[%s] done, memory use %s', self._timer, util.max_memory())

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
        if self._use_mkl and _mkl_ops is not None:
            return self._mkl_similarities(rmat)
        else:
            return self._scipy_similarities(rmat)

    def _scipy_similarities(self, rmat):
        sp_rmat = rmat.to_scipy()

        _logger.info('[%s] multiplying matrix with scipy', self._timer)
        smat = sp_rmat.T @ sp_rmat
        smat = matrix.CSR.from_scipy(smat, False)

        csr = self._filter_select(smat)
        return csr

    def _mkl_similarities(self, rmat):
        assert rmat.values is not None

        _logger.info('[%s] multiplying matrix with MKL', self._timer)
        m_nbrs = self.save_nbrs
        if m_nbrs is None or m_nbrs < 0:
            m_nbrs = 0
        trmat = rmat.transpose()
        nitems = trmat.nrows

        # for i in range(nitems):
        #     _logger.debug('verifying row %d', i)
        #     cs = trmat.row_cs(i)
        #     assert np.all(cs >= 0)
        #     assert np.all(cs < trmat.ncols)
        #     assert pd.Series(cs).nunique() == len(cs)

        _logger.debug('[%s] transposed, memory use %s', self._timer, util.max_memory())
        s_blocks = _mkl_sim_blocks(trmat.N, self.min_sim, m_nbrs)
        _logger.debug('[%s] computed blocks, memory use %s', self._timer, util.max_memory())
        s_blocks = [matrix.CSR(N=b) for (b, bs, be) in s_blocks]
        nnz = sum(b.nnz for b in s_blocks)
        tot_rows = sum(b.nrows for b in s_blocks)
        _logger.info('[%s] computed %d similarities for %d items in %d blocks',
                     self._timer, nnz, tot_rows, len(s_blocks))
        row_nnzs = np.concatenate([b.row_nnzs() for b in s_blocks])
        assert len(row_nnzs) == nitems, \
            'only have {} rows for {} items'.format(len(row_nnzs), nitems)

        smat = matrix.CSR.empty((nitems, nitems), row_nnzs, rpdtype=np.int64)
        start = 0
        for bi, b in enumerate(s_blocks):
            bnr = b.nrows
            end = start + bnr
            v_sp = smat.rowptrs[start]
            v_ep = smat.rowptrs[end]
            _logger.debug('block %d (%d:%d) has %d entries, storing in %d:%d',
                          bi, start, end, b.nnz, v_sp, v_ep)
            smat.colinds[v_sp:v_ep] = b.colinds
            smat.values[v_sp:v_ep] = b.values
            start = end

        _logger.info('[%s] sorting similarity matrix with %d entries', self._timer, smat.nnz)
        _sort_nbrs(smat.N)

        return smat

    def _filter_select(self, smat):
        "Threshold, filter, and symmetrify matrices"
        nitems = smat.nrows
        assert smat.ncols == nitems

        # Count possible neighbors
        _logger.debug('counting neighbors')
        possible = _count_nbrs(smat.N, self.min_sim)

        # Count neighbors to use neighbors
        save_nbrs = self.save_nbrs
        if save_nbrs is not None and save_nbrs > 0:
            nnbrs = np.minimum(possible, save_nbrs, dtype=np.int32)
        else:
            nnbrs = possible
        nsims = np.sum(nnbrs)

        # set up the target matrix
        _logger.info('[%s] truncating %d neighbors to %d (of %d possible)',
                     self._timer, smat.nnz, nsims, np.sum(possible))
        trimmed = matrix.CSR.empty((nitems, nitems), nnbrs)

        # copy values into target arrays
        used = _copy_nbrs(smat.N, trimmed.N, nnbrs, self.min_sim)
        assert np.all(used == nnbrs)

        _logger.info('[%s] sorting neighborhoods', self._timer)
        _sort_nbrs(trimmed.N)

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
        n_items = len(self.item_index_)
        ri_pos = self.item_index_.get_indexer(ratings.index)
        m_rates = ratings[ri_pos >= 0]
        ri_pos = ri_pos[ri_pos >= 0]
        rate_v = np.full(n_items, np.nan, dtype=np.float_)
        rated = np.zeros(n_items, dtype='bool')
        # mean-center the rating array
        if self.center:
            rate_v[ri_pos] = m_rates.values - self.item_means_[ri_pos]
        else:
            rate_v[ri_pos] = m_rates.values
        rated[ri_pos] = True

        _logger.debug('user %s: %d of %d rated items in model', user, len(ri_pos), len(ratings))
        assert np.sum(np.logical_not(np.isnan(rate_v))) == len(ri_pos)
        assert np.all(np.isnan(rate_v) == np.logical_not(rated))

        # set up item result vector
        # ipos will be an array of item indices
        i_pos = self.item_index_.get_indexer(items)
        i_pos = i_pos[i_pos >= 0]
        _logger.debug('user %s: %d of %d requested items in model', user, len(i_pos), len(items))

        # now we take a first pass through the data to count _viable_ targets
        # This computes the number of neighbors (and their weight sum) for
        # each target item based on the user's ratings, allowing us to fast-path
        # other computations and avoid as many neighbor truncations as possible
        i_cts, i_sums, i_nbrs = self._count_viable_targets(i_pos, ri_pos)
        viable = i_cts >= self.min_nbrs
        i_pos = i_pos[viable]
        i_cts = i_cts[viable]
        i_sums = i_sums[viable]
        i_nbrs = i_nbrs[viable]
        _logger.debug('user %s: %d of %d requested items possibly reachable',
                      user, len(i_pos), len(items))

        # look for some fast paths
        if self.aggregate == self.AGG_SUM and self.min_sim >= 0:
            # similarity sums are all we need
            if self.nnbrs >= 0:
                fast_mask = i_cts <= self.nnbrs
                fast_items = i_pos[fast_mask]
                fast_scores = i_sums[fast_mask]
                slow_items = i_pos[~fast_mask]
            else:
                fast_items = i_pos
                fast_scores = i_sums
                slow_items = np.array([], dtype='i4')

            _logger.debug('user %s: using fast-path similarity sum for %d items',
                          user, len(fast_items))

            if len(slow_items):
                iscores = _predict_sum(self.sim_matrix_.N, len(self.item_index_),
                                       (self.min_nbrs, self.nnbrs),
                                       rate_v, rated, slow_items)
            else:
                iscores = np.full(len(self.item_index_), np.nan)
            iscores[fast_items] = fast_scores

        elif self.aggregate == self.AGG_WA and self.min_nbrs == 1:
            # fast-path single-neighbor targets - common in sparse data
            fast_mask = i_cts == 1
            fast_items = i_pos[fast_mask]
            fast_scores = rate_v[i_nbrs[fast_mask]]
            if self.min_sim < 0:
                fast_scores *= np.sign(i_sums[fast_mask])
            _logger.debug('user %s: fast-pathed %d scores', user, len(fast_scores))

            slow_items = i_pos[i_cts > 1]
            iscores = _predict_weighted_average(self.sim_matrix_.N, len(self.item_index_),
                                                (self.min_nbrs, self.nnbrs),
                                                rate_v, rated, slow_items)
            iscores[fast_items] = fast_scores
        else:
            # now compute the predictions
            _logger.debug('user %s: taking the slow path', user)
            agg = _predictors[self.aggregate]
            iscores = agg(self.sim_matrix_.N, len(self.item_index_), (self.min_nbrs, self.nnbrs),
                          rate_v, rated, i_pos)

        if self.center:
            iscores += self.item_means_

        results = pd.Series(iscores, index=self.item_index_)
        results = results.reindex(items, fill_value=np.nan)

        _logger.debug('user %s: predicted for %d of %d items',
                      user, results.notna().sum(), len(items))

        return results

    def _count_viable_targets(self, targets, rated):
        "Count upper-bound on possible neighbors for target items and rated items."
        # initialize counts to zero
        counts = np.zeros(len(self.item_index_), dtype=np.int32)
        sums = np.zeros(len(self.item_index_))
        last_nbrs = np.full(len(self.item_index_), -1, 'i4')
        # count the number of times each item is reachable from the neighborhood
        for ri in rated:
            nbrs = self._sim_inv_.row_cs(ri)
            counts[nbrs] += 1
            sums[nbrs] += self._sim_inv_.row_vs(ri)
            last_nbrs[nbrs] = ri

        # we want the reachability counts for the target items
        return counts[targets], sums[targets], last_nbrs[targets]

    def __getstate__(self):
        state = dict(self.__dict__)
        if '_sim_inv_' in state:
            del state['_sim_inv_']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, 'sim_matrix_'):
            self._sim_inv_ = self.sim_matrix_.transpose()

    def __str__(self):
        return 'ItemItem(nnbrs={}, msize={})'.format(self.nnbrs, self.save_nbrs)
