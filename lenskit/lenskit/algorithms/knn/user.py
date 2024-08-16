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

import numpy as np
import pandas as pd
import torch
from typing_extensions import NamedTuple, Optional, Self, override

from lenskit import util
from lenskit.data import FeedbackType
from lenskit.data.dataset import Dataset
from lenskit.data.vocab import Vocabulary
from lenskit.diagnostics import DataWarning
from lenskit.math.sparse import normalize_sparse_rows, safe_spmv
from lenskit.parallel.config import ensure_parallel_init

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

    users_: Vocabulary
    "The index of user IDs."
    items_: Vocabulary
    "The index of item IDs."
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

        if self.aggregate not in [self.AGG_WA, self.AGG_SUM]:
            raise ValueError(f"invalid aggregate {self.aggregate}")

    @override
    def fit(self, data: Dataset, **kwargs) -> Self:
        """
        "Train" a user-user CF model.  This memorizes the rating data in a format that is usable
        for future computations.

        Args:
            ratings(pandas.DataFrame): (user, item, rating) data for collaborative filtering.
        """
        ensure_parallel_init()
        rmat = data.interaction_matrix("torch", field="rating" if self.use_ratings else None)
        assert rmat.is_sparse_csr

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
        self.user_ratings_ = rmat.to_sparse_coo().coalesce()
        self.users_ = data.users.copy()
        self.user_means_ = means
        self.items_ = data.items.copy()

        return self

    @override
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
        items = np.asarray(items)

        udata = self._get_user_data(user, ratings)
        if udata is None:
            _log.debug("user %s has no ratings, skipping", user)
            return pd.Series(index=items, dtype="float32")

        uidx, ratings, umean = udata
        _log.debug("scoring %d items for user %s (idx %d)", len(items), user, uidx or -1)
        assert ratings.shape == (len(self.items_),)  # ratings is a dense vector

        # now ratings has vbeen normalized to be a mean-centered unit vector
        # this means we can dot product to score neighbors
        # score the neighbors!
        nbr_sims = safe_spmv(self.user_vectors_, ratings)
        assert nbr_sims.shape == (len(self.users_),)
        if uidx is not None:
            # zero out the self-similarity
            nbr_sims[uidx] = 0

        # get indices for these neighbors
        nbr_idxs = torch.arange(len(self.users_), dtype=torch.int64)

        nbr_mask = nbr_sims >= self.min_sim

        kn_sims = nbr_sims[nbr_mask]
        kn_idxs = nbr_idxs[nbr_mask]
        if len(kn_sims) > 0:
            _log.debug(
                "user %s: %d candidate neighbors (of %d total), max sim %0.4f",
                user,
                len(kn_sims),
                len(self.users_),
                torch.max(kn_sims).item(),
            )
        else:
            _log.warn("user %s: no candidate neighbors", user)
            return pd.Series(index=items, dtype="float32")

        assert not torch.any(torch.isnan(kn_sims))

        iidxs = self.items_.numbers(items, missing="negative")
        iidxs = torch.from_numpy(iidxs).to(torch.int64)

        ki_mask = iidxs >= 0
        usable_iidxs = iidxs[ki_mask]

        scores = score_items_with_neighbors(
            usable_iidxs,
            kn_idxs,
            kn_sims,
            self.user_ratings_,
            self.nnbrs,
            self.min_nbrs,
            self.aggregate == "weighted-average",
        )

        scores += umean

        results = pd.Series(scores.numpy(), index=items[ki_mask.numpy()], name="prediction")
        results = results.reindex(items)

        _log.debug(
            "scored %d of %d items for %s in %s", results.notna().sum(), len(items), user, watch
        )
        return results

    def _get_user_data(self, user, ratings) -> Optional[UserRatings]:
        "Get a user's data for user-user CF"

        index = self.users_.number(user, missing=None)

        if ratings is None:
            if index is None:
                _log.warning("user %d has no ratings and none provided", user)
                return None

            assert index >= 0
            row = self.user_vectors_[index].to_dense()
            if self.center:
                assert self.user_means_ is not None
                umean = self.user_means_[index].item()
            else:
                umean = 0
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
            ratings = ratings.reindex(self.items_, fill_value=0).values
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


def score_items_with_neighbors(
    items: torch.Tensor,
    nbr_rows: torch.Tensor,
    nbr_sims: torch.Tensor,
    ratings: torch.Tensor,
    max_nbrs: int,
    min_nbrs: int,
    average: bool,
) -> torch.Tensor:
    # select a sub-matrix for further manipulation
    (ni,) = items.shape
    nbr_rates = ratings.index_select(0, nbr_rows)
    nbr_rates = nbr_rates.index_select(1, items)
    nbr_rates = nbr_rates.coalesce()
    assert nbr_rates.shape == (
        len(nbr_rows),
        ni,
    ), f"nbr rates has shape {nbr_rates.shape} (expected {len(nbr_rows)}, {ni})"
    nbr_t = nbr_rates.transpose(0, 1).to_sparse_csr()

    # count nbrs for each item
    counts = nbr_t.crow_indices().diff()
    assert counts.shape == items.shape

    _log.debug(
        "scoring %d items, max count %d, nbr shape %s",
        ni,
        torch.max(counts).item(),
        nbr_rates.shape,
    )

    # fast-path items with small neighborhoods
    fp_mask = counts <= max_nbrs
    r_mask = fp_mask[nbr_rates.indices()[1]]
    nbr_fp = torch.sparse_coo_tensor(
        indices=nbr_rates.indices()[:, r_mask],
        values=nbr_rates.values()[r_mask],
        size=nbr_rates.shape,
    ).coalesce()
    results = torch.mv(nbr_fp.transpose(0, 1), nbr_sims)

    if average:
        nnz = len(nbr_fp.values())
        nbr_fp_ones = torch.sparse_coo_tensor(
            indices=nbr_fp.indices(),
            values=torch.ones(nnz),
            size=nbr_rates.shape,
        )
        tot_sims = torch.mv(nbr_fp_ones.transpose(0, 1), nbr_sims)
        assert torch.all(torch.isfinite(tot_sims))
        results /= tot_sims

    # clear out too-small neighborhoods
    results[counts < min_nbrs] = torch.nan

    # deal with too-large items
    exc_mask = counts > max_nbrs
    if torch.any(exc_mask):
        _log.debug("scoring %d slow-path items", torch.sum(exc_mask))

    bads = torch.argwhere(exc_mask)[:, 0]
    for badi in bads:
        col = nbr_t[badi]
        assert col.shape == nbr_rates.shape[:1]

        bi_users = col.indices()[0]
        bi_rates = col.values()
        bi_sims = nbr_sims[bi_users]

        tk_vs, tk_is = torch.topk(bi_sims, max_nbrs)
        sum = torch.sum(tk_vs)
        if average:
            results[badi] = torch.dot(tk_vs, bi_rates[tk_is]) / sum
        else:
            results[badi] = sum

    return results
