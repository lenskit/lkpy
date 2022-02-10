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
from csr import CSR, create_from_sizes, create_empty
import csr.kernel as csrk
from numba import njit, prange
from numba.typed import List

from lenskit import util, DataWarning, ConfigWarning
from lenskit.data import sparse_ratings
from lenskit.sharing import in_share_context
from lenskit.util.parallel import is_mp_worker
from lenskit.util.accum import kvp_minheap_insert, kvp_minheap_sort
from . import Predictor

_logger = logging.getLogger(__name__)


def _make_blocks(n, size):
    "Create blocks for the range 0..n."
    return [(s, min(s + size, n)) for s in range(0, n, size)]


@njit(parallel=not is_mp_worker())
def _sort_nbrs(smat: CSR):
    for i in prange(smat.nrows):
        sp, ep = smat.row_extent(i)
        kvp_minheap_sort(sp, ep, smat.colinds, smat.values)


@njit
def _trim_sim_block(nitems, bsp, bitems, block, min_sim, max_nbrs):
    # pass 1: compute the size of each row
    sizes = np.zeros(bitems, np.int32)
    for i in range(nitems):
        sp, ep = block.row_extent(i)
        for j in range(sp, ep):
            # we accept the neighbor if it passes threshold and isn't a self-similarity
            r = block.colinds[j]
            if i != bsp + r and block.values[j] >= min_sim:
                sizes[r] += 1

    if max_nbrs > 0:
        for i in range(bitems):
            if sizes[i] > max_nbrs:
                sizes[i] = max_nbrs

    # if bnc == 0:
    #     # empty resulting matrix, oops
    #     return _empty_csr(bitems, nitems, np.zeros(bitems, np.int32))

    # allocate a matrix
    block_csr = create_from_sizes(bitems, nitems, sizes)

    # pass 2: truncate each row into the matrix
    eps = block_csr.rowptrs[:-1].copy()
    for c in range(nitems):
        sp, ep = block.row_extent(c)
        for j in range(sp, ep):
            v = block.values[j]
            r = block.colinds[j]
            sp, lep = block_csr.row_extent(r)
            lim = lep - sp
            if c != bsp + r and v >= min_sim:
                eps[r] = kvp_minheap_insert(sp, eps[r], lim, c, v,
                                            block_csr.colinds, block_csr.values)
        # we're done!
    return block_csr


@njit
def _sim_block(block, bsp, bep, rmh, min_sim, max_nbrs, nitems):
    "Compute a single block of the similarity matrix"
    # assert block.nrows == bep - bsp

    bitems = block.nrows
    if block.nnz == 0:
        return create_empty(bitems, nitems)

    # create a matrix handle for the subset matrix
    amh = csrk.to_handle(block)

    smh = csrk.mult_abt(rmh, amh)

    csrk.release_handle(amh)

    csrk.order_columns(smh)
    block_csr = csrk.from_handle(smh)
    # this is allowed now
    csrk.release_handle(smh)

    block_csr = _trim_sim_block(nitems, bsp, bitems, block_csr, min_sim, max_nbrs)

    return block_csr


@njit(nogil=True, parallel=not is_mp_worker())
def _sim_blocks(trmat, blocks, ptrs, min_sim, max_nbrs):
    "Compute the similarity matrix with blocked CSR kernel calls"
    nitems = trmat.nrows
    nblocks = len(blocks)

    null = create_empty(1, 1)
    res = [null for i in range(nblocks)]

    rmat_h = csrk.to_handle(trmat)
    csrk.order_columns(rmat_h)

    for bi in prange(nblocks):
        b = blocks[bi]
        p = ptrs[bi]
        bs, be = p
        bres = _sim_block(b, bs, be, rmat_h, min_sim, max_nbrs, nitems)
        res[bi] = bres

    csrk.release_handle(rmat_h)

    return res


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
    Java-based LensKit code :cite:p:`Ekstrand2011-bp`.  This implementation is based on the description
    of item-based CF by :cite:t:`Deshpande2004-ht`, and produces results equivalent to Java LensKit.

    The k-NN predictor supports several aggregate functions:

    ``weighted-average``
        The weighted average of the user's rating values, using item-item similarities as
        weights.

    ``sum``
        The sum of the similarities between the target item and the user's rated items,
        regardless of the rating the user gave the items.

    Args:
        nnbrs(int):
            the maximum number of neighbors for scoring each item (``None`` for unlimited)
        min_nbrs(int): the minimum number of neighbors for scoring each item
        min_sim(float): minimum similarity threshold for considering a neighbor
        save_nbrs(float):
            the number of neighbors to save per item in the trained model
            (``None`` for unlimited)
        feedback(str):
            Control how feedback should be interpreted.  Specifies defaults for the other
            settings, which can be overridden individually; can be one of the following values:

            ``explicit``
                Configure for explicit-feedback mode: use rating values, center ratings, and
                use the ``weighted-average`` aggregate method for prediction.  This is the
                default setting.

            ``implicit``
                Configure for implicit-feedback mode: ignore rating values, do not center ratings,
                and use the ``sum`` aggregate method for prediction.
        center(bool):
            whether to normalize (mean-center) rating vectors prior to computing similarities
            and aggregating user rating values.  Defaults to ``True``; turn this off when working
            with unary data and other data types that don't respond well to centering.
        aggregate(str):
            the type of aggregation to do. Can be ``weighted-average`` (the default) or ``sum``.
        use_ratings(bool):
            whether or not to use the rating values. If ``False``, it ignores rating values and
            considers an implicit feedback signal of 1 for every (user,item) pair present.

    Attributes:
        item_index_(pandas.Index): the index of item IDs.
        item_means_(numpy.ndarray): the mean rating for each known item.
        item_counts_(numpy.ndarray): the number of saved neighbors for each item.
        sim_matrix_(matrix.CSR): the similarity matrix.
        user_index_(pandas.Index): the index of known user IDs for the rating matrix.
        rating_matrix_(matrix.CSR): the user-item rating matrix for looking up users' ratings.
    """
    IGNORED_PARAMS = ['feedback']
    EXTRA_PARAMS = ['center', 'aggregate', 'use_ratings']

    AGG_SUM = intern('sum')
    AGG_WA = intern('weighted-average')
    RATING_AGGS = [AGG_WA]  # the aggregates that use rating values

    def __init__(self, nnbrs, min_nbrs=1, min_sim=1.0e-6, save_nbrs=None, feedback='explicit',
                 **kwargs):
        self.nnbrs = nnbrs
        if self.nnbrs is not None and self.nnbrs < 1:
            self.nnbrs = -1
        self.min_nbrs = min_nbrs
        if self.min_nbrs is not None and self.min_nbrs < 1:
            self.min_nbrs = 1
        self.min_sim = min_sim
        self.save_nbrs = save_nbrs

        if feedback == 'explicit':
            defaults = {
                'center': True,
                'aggregate': self.AGG_WA,
                'use_ratings': True
            }
        elif feedback == 'implicit':
            defaults = {
                'center': False,
                'aggregate': self.AGG_SUM,
                'use_ratings': False
            }
        else:
            raise ValueError(f'invalid feedback mode: {feedback}')

        defaults.update(kwargs)
        self.center = defaults['center']
        self.aggregate = intern(defaults['aggregate'])
        self.use_ratings = defaults['use_ratings']

        self._check_setup()

    def _check_setup(self):
        if not self.use_ratings:
            if self.center:
                _logger.warning('item-item configured to ignore ratings, but ``center=True`` - likely bug')
                warnings.warn(util.clean_str('''
                    item-item configured to ignore ratings, but ``center=True``.  This configuration
                    is unlikely to work well.
                '''), ConfigWarning)
            if self.aggregate == 'weighted-average':
                _logger.warning('item-item configured to ignore ratings, but using weighted averages - likely bug')
                warnings.warn(util.clean_str('''
                    item-item configured to ignore ratings, but use weighted averages.  This configuration
                    is unlikely to work well.
                '''), ConfigWarning)

    def fit(self, ratings, **kwargs):
        """
        Train a model.

        The model-training process depends on ``save_nbrs`` and ``min_sim``, but *not* on other
        algorithm parameters.

        Args:
            ratings(pandas.DataFrame):
                (user,item,rating) data for computing item similarities.
        """
        util.check_env()
        # Training proceeds in 2 steps:
        # 1. Normalize item vectors to be mean-centered and unit-normalized
        # 2. Compute similarities with pairwise dot products
        self._timer = util.Stopwatch()

        _logger.debug('[%s] beginning fit, memory use %s', self._timer, util.max_memory())
        _logger.debug('[%s] using CSR kernel %s', self._timer, csrk.name)

        init_rmat, users, items = sparse_ratings(ratings)
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
        nmat = rmat.copy(False)
        nmat.values = mcvals
        if np.allclose(nmat.values, 0):
            _logger.warn('normalized ratings are zero, centering is not recommended')
            warnings.warn("Ratings seem to have the same value, centering is not recommended.",
                          DataWarning)
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
        return CSR.from_scipy(norm_mat, False)

    def _compute_similarities(self, rmat):
        trmat = rmat.transpose()
        nitems = trmat.nrows
        m_nbrs = self.save_nbrs
        if m_nbrs is None or m_nbrs < 0:
            m_nbrs = 0

        bounds = _make_blocks(nitems, 1000)
        _logger.info('[%s] splitting %d items (%d ratings) into %d blocks',
                     self._timer, nitems, trmat.nnz, len(bounds))
        blocks = [trmat.subset_rows(sp, ep) for (sp, ep) in bounds]

        _logger.info('[%s] computing similarities', self._timer)
        ptrs = List(bounds)
        nbs = List(blocks)
        if not nbs:
            # oops, this is the bad place
            # in non-JIT node, List doesn't actually make the list
            nbs = blocks
            ptrs = bounds
        s_blocks = _sim_blocks(trmat, nbs, ptrs, self.min_sim, m_nbrs)

        nnz = sum(b.nnz for b in s_blocks)
        tot_rows = sum(b.nrows for b in s_blocks)
        _logger.info('[%s] computed %d similarities for %d items in %d blocks',
                     self._timer, nnz, tot_rows, len(s_blocks))
        row_nnzs = np.concatenate([b.row_nnzs() for b in s_blocks])
        assert len(row_nnzs) == nitems, \
            'only have {} rows for {} items'.format(len(row_nnzs), nitems)

        smat = CSR.empty(nitems, nitems, row_nnzs)
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
        _sort_nbrs(smat)

        return smat

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
                iscores = _predict_sum(self.sim_matrix_, len(self.item_index_),
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
            iscores = _predict_weighted_average(self.sim_matrix_, len(self.item_index_),
                                                (self.min_nbrs, self.nnbrs),
                                                rate_v, rated, slow_items)
            iscores[fast_items] = fast_scores
        else:
            # now compute the predictions
            _logger.debug('user %s: taking the slow path', user)
            agg = _predictors[self.aggregate]
            iscores = agg(self.sim_matrix_, len(self.item_index_), (self.min_nbrs, self.nnbrs),
                          rate_v, rated, i_pos)

        if self.center and self.aggregate in self.RATING_AGGS:
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
        if '_sim_inv_' in state and not in_share_context():
            del state['_sim_inv_']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, 'sim_matrix_') and not hasattr(self, '_sim_inv_'):
            self._sim_inv_ = self.sim_matrix_.transpose()

    def __str__(self):
        return 'ItemItem(nnbrs={}, msize={})'.format(self.nnbrs, self.save_nbrs)
