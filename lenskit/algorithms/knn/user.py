# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
User-based k-NN collaborative filtering.
"""

# pyright: basic
from __future__ import annotations

import logging
import warnings
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
from numba import njit

from lenskit import DataWarning, util
from lenskit.data import FeedbackType, sparse_ratings
from lenskit.data.matrix import normalize_sparse_rows, sparse_row_stats
from lenskit.util.accum import kvp_minheap_insert

from .. import Predictor

_log = logging.getLogger(__name__)


class UserUser(Predictor):
    """
    User-user nearest-neighbor collaborative filtering with ratings. This user-user implementation
    is not terribly configurable; it hard-codes design decisions found to work well in the previous
    Java-based LensKit code.

    Args:
        nnbrs:
            the maximum number of neighbors for scoring each item (``None`` for unlimited).
        min_nbrs:
            The minimum number of neighbors for scoring each item.
        min_sim:
            Minimum similarity threshold for considering a neighbor.  Must be
            positive; if less than the smallest 32-bit normal (:math:`1.175
            \\times 10^{-38}`), is clamped to that value.
        feedback:
            Control how feedback should be interpreted.  Specifies defaults for the other
            settings, which can be overridden individually; can be one of the following values:

            ``explicit``
                Configure for explicit-feedback mode: use rating values, center ratings, and
                use the ``weighted-average`` aggregate method for prediction.  This is the
                default setting.

            ``implicit``
                Configure for implicit-feedback mode: ignore rating values, do not center ratings,
                and use the ``sum`` aggregate method for prediction.
        center:
            whether to normalize (mean-center) rating vectors.  Turn this off when working
            with unary data and other data types that don't respond well to centering.
        aggregate:
            the type of aggregation to do. Can be ``weighted-average`` or ``sum``.
        use_ratings:
            whether or not to use rating values; default is ``True``.  If ``False``, it ignores
            rating values and treates every present rating as 1.
    """

    IGNORED_PARAMS = ["feedback"]
    EXTRA_PARAMS = ["center", "aggregate", "use_ratings"]
    AGG_SUM = "sum"
    AGG_WA = "weighted-average"
    RATING_AGGS = [AGG_WA]

    nnbrs: int
    min_nbrs: int
    min_sim: float
    feedback: FeedbackType
    center: bool
    aggregate: str
    use_ratings: bool

    user_index_: pd.Index
    "The index of user IDs."
    user_means_: torch.Tensor | None
    "Mean rating for each known user."
    user_vectors_: torch.Tensor
    "Normalized rating matrix (CSR) to find neighbors at prediction time."
    user_ratings_: torch.Tensor
    "Centered but un-normalized rating matrix (COO) to find neighbor ratings."

    def __init__(
        self,
        nnbrs: int,
        min_nbrs: int = 1,
        min_sim: float = 1.0e-6,
        feedback: FeedbackType = "explicit",
        **kwargs,
    ):
        self.nnbrs = nnbrs
        self.min_nbrs = min_nbrs
        if min_sim < 0:
            raise ValueError("minimum similarity must be positive")
        elif min_sim == 0:
            f4i = np.finfo("f4")
            self.min_sim = float(f4i.smallest_normal)
            _log.warn("minimum similarity %e is too low, using %e", min_sim, self.min_sim)
        else:
            self.min_sim = min_sim

        if feedback == "explicit":
            defaults = {"center": True, "aggregate": self.AGG_WA, "use_ratings": True}
        elif feedback == "implicit":
            defaults = {"center": False, "aggregate": self.AGG_SUM, "use_ratings": False}
        else:
            raise ValueError(f"invalid feedback mode: {feedback}")

        defaults.update(kwargs)
        self.center = defaults["center"]
        self.aggregate = defaults["aggregate"]
        self.use_ratings = defaults["use_ratings"]

    def fit(self, ratings, **kwargs):
        """
        "Train" a user-user CF model.  This memorizes the rating data in a format that is usable
        for future computations.

        Args:
            ratings(pandas.DataFrame): (user, item, rating) data for collaborative filtering.
        """
        util.check_env()
        rmat, users, items = sparse_ratings(ratings, torch=True)

        if self.center:
            rmat, means = normalize_sparse_rows(rmat, "center")
            if np.allclose(rmat.values(), 0.0):
                _log.warn("normalized ratings are zero, centering is not recommended")
                warnings.warn(
                    "Ratings seem to have the same value, centering is not recommended.",
                    DataWarning,
                )
        else:
            means = None

        normed, _norms = normalize_sparse_rows(rmat, "unit")

        self.user_vectors_ = normed
        self.user_ratings_ = rmat.to_sparse_coo()
        self.user_index_ = users
        self.user_means_ = means
        self.item_index_ = items

        return self

    def predict_for_user(self, user, items, ratings=None):
        """
        Compute predictions for a user and items.

        Args:
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series):
                the user's ratings (indexed by item id); if provided, will be used to
                recompute the user's bias at prediction time.

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """

        watch = util.Stopwatch()
        items = pd.Index(items, name="item")

        udata = self._get_user_data(user, ratings)
        if udata is None:
            _log.debug("user %s has no ratings, skipping", user)
            return pd.Series(index=items, dtype="float32")

        index, ratings, umean = udata
        assert len(ratings) == len(self.item_index_)  # ratings is a dense vector

        # now ratings is normalized to be a mean-centered unit vector
        # this means we can dot product to score neighbors
        # score the neighbors!
        nbr_sims = torch.mv(self.user_vectors_, ratings)
        assert nbr_sims.shape == (len(self.user_index_),)
        if index is not None:
            # zero out the self-similarity
            nbr_sims[index] = 0

        # get indices for these neighbors
        nbr_idxs = torch.arange(len(self.user_index_), dtype=torch.int64)

        nbr_mask = nbr_sims >= self.min_sim

        kn_sims = nbr_sims[nbr_mask]
        kn_idxs = nbr_idxs[nbr_mask]

        _log.debug("found %d candidate neighbor similarities", kn_sims.shape[0])

        iidxs = self.item_index_.get_indexer(items.values)
        iidxs = torch.from_numpy(iidxs)
        if self.aggregate == self.AGG_WA:
            agg = _agg_weighted_avg
        elif self.aggregate == self.AGG_SUM:
            agg = _agg_sum
        else:
            raise ValueError("invalid aggregate " + self.aggregate)

        scores = score_items_with_neighbors(
            iidxs,
            kn_idxs,
            kn_sims,
            self.user_ratings_,
            self.nnbrs,
            self.min_nbrs,
            agg,
        )

        scores += umean

        results = pd.Series(scores, index=items, name="prediction")

        _log.debug(
            "scored %d of %d items for %s in %s", results.notna().sum(), len(items), user, watch
        )
        return results

    def _get_user_data(self, user, ratings) -> Optional[UserRatings]:
        "Get a user's data for user-user CF"

        try:
            index = self.user_index_.get_loc(user)
            assert isinstance(index, int)
        except KeyError:
            index = None

        if ratings is None:
            if index is None:
                _log.warning("user %d has no ratings and none provided", user)
                return None

            row = self.user_vectors_[index].to_dense()
            umean = self.user_means_[index].item() if self.user_means_ is not None else 0
            return UserRatings(index, row, umean)
        else:
            _log.debug("using provided ratings for user %d", user)
            if self.center:
                umean = ratings.mean()
                ratings = ratings - umean
            else:
                umean = 0
            unorm = np.linalg.norm(ratings)
            ratings = ratings / unorm
            ratings = ratings.reindex(self.item_index_, fill_value=0).values
            ratings = torch.from_numpy(np.require(ratings, "f4"))
            return UserRatings(index, ratings, umean)

    def __str__(self):
        return "UserUser(nnbrs={}, min_sim={})".format(self.nnbrs, self.min_sim)


class UserRatings(NamedTuple):
    """
    Dense user ratings.
    """

    index: int | None
    ratings: torch.Tensor
    mean: float


@njit
def _agg_weighted_avg(iur, item, sims, use):
    """
    Weighted-average aggregate.

    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    rates = iur.row_vs(item)
    num = 0.0
    den = 0.0
    for j in use:
        num += rates[j] * sims[j]
        den += np.abs(sims[j])
    return num / den


@njit
def _agg_sum(iur, item, sims, use):
    """
    Sum aggregate

    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    x = 0.0
    for j in use:
        x += sims[j]
    return x


def score_items_with_neighbors(
    items: torch.Tensor,
    nbr_rows: torch.Tensor,
    nbr_sims: torch.Tensor,
    ratings: torch.Tensor,
    max_nbrs: int,
    min_nbrs: int,
    agg,
) -> torch.Tensor:
    # select a sub-matrix for further manipulation
    (ni,) = items.shape
    nbr_rates = ratings.index_select(0, nbr_rows)
    nbr_rates = nbr_rates.index_select(1, items)
    nbr_rates = nbr_rates.coalesce()
    nbr_t = nbr_rates.transpose(0, 1).to_sparse_csr()

    # count nbrs for each item
    counts = nbr_t.crow_indices().diff()
    assert counts.shape == items.shape

    _log.debug("scoring %d items, max count %d", ni, torch.max(counts).item())

    # fast-path items with small neighborhoods
    fp_mask = counts <= max_nbrs
    r_mask = fp_mask[nbr_rates.indices()[1]]
    nbr_fp = torch.sparse_coo_tensor(
        indices=nbr_rates.indices()[:, r_mask],
        values=nbr_rates.values()[r_mask],
        size=nbr_rates.shape,
    ).coalesce()
    results = torch.mv(nbr_fp.transpose(0, 1), nbr_sims)

    nnz = len(nbr_fp.values())
    nbr_fp_ones = torch.sparse_coo_tensor(
        indices=nbr_rates.indices()[:, r_mask],
        values=torch.ones(nnz),
        size=nbr_rates.shape,
    )
    results /= torch.mv(nbr_fp_ones.transpose(0, 1), nbr_sims)

    # clear out too-small neighborhoods
    results[counts < min_nbrs] = torch.nan

    # deal with too-large items
    exc_mask = counts > max_nbrs
    if torch.any(exc_mask):
        _log.debug("scoring %d slow-path items", torch.sum(exc_mask))
    for badi in items[exc_mask]:
        col = nbr_t[badi]
        assert col.shape == nbr_rates.shape[:1]

        bi_users = col.indices()[0]
        bi_rates = col.values()
        bi_sims = nbr_sims[bi_users]

        tk_vs, tk_is = torch.topk(bi_sims, max_nbrs)
        results[badi] = torch.sum(tk_vs * bi_rates[tk_is])

    return results


@njit
def _score(items, results, iur, sims, nnbrs, min_sim, min_nbrs, agg):
    h_ks = np.empty(nnbrs, dtype=np.int32)
    h_vs = np.empty(nnbrs)
    used = np.zeros(len(results), dtype=np.int32)

    for i in range(len(results)):
        item = items[i]
        if item < 0:
            continue

        h_ep = 0

        # who has rated this item?
        i_users = iur.row_cs(item)

        # what are their similarities to our target user?
        i_sims = sims[i_users]

        # which of these neighbors do we really want to use?
        for j, s in enumerate(i_sims):
            if np.abs(s) < 1.0e-10:
                continue
            if min_sim is not None and s < min_sim:
                continue
            h_ep = kvp_minheap_insert(0, h_ep, nnbrs, j, s, h_ks, h_vs)

        if h_ep < min_nbrs:
            continue

        results[i] = agg(iur, item, i_sims, h_ks[:h_ep])
        used[i] = h_ep

    return used
