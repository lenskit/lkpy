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

import warnings

import numpy as np
import pandas as pd
import structlog
import torch
from pydantic import BaseModel, PositiveFloat, PositiveInt, field_validator
from scipy.sparse import csc_array
from typing_extensions import NamedTuple, Optional, override

from lenskit import util
from lenskit.data import Dataset, FeedbackType, ItemList, QueryInput, RecQuery
from lenskit.data.vocab import Vocabulary
from lenskit.diagnostics import DataWarning
from lenskit.logging import get_logger
from lenskit.math.sparse import normalize_sparse_rows, safe_spmv, torch_sparse_to_scipy
from lenskit.parallel.config import ensure_parallel_init
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_log = get_logger(__name__)


class UserKNNConfig(BaseModel, extra="forbid"):
    "Configuration for :class:`ItemKNNScorer`."

    k: PositiveInt = 20
    """
    The maximum number of neighbors for scoring each item.
    """
    min_nbrs: PositiveInt = 1
    """
    The minimum number of neighbors for scoring each item.
    """
    min_sim: PositiveFloat = 1.0e-6
    """
    Minimum similarity threshold for considering a neighbor.  Must be positive;
    if less than the smallest 32-bit normal (:math:`1.175 \\times 10^{-38}`), is
    clamped to that value.
    """
    feedback: FeedbackType = "explicit"
    """
    The type of input data to use (explicit or implicit).  This affects data
    pre-processing and aggregation.
    """

    @field_validator("min_sim", mode="after")
    @staticmethod
    def clamp_min_sim(sim) -> float:
        return max(sim, float(np.finfo(np.float64).smallest_normal))

    @property
    def explicit(self) -> bool:
        """
        Query whether this is in explicit-feedback mode.
        """
        return self.feedback == "explicit"


class UserKNNScorer(Component[ItemList], Trainable):
    """
    User-user nearest-neighbor collaborative filtering with ratings. This
    user-user implementation is not terribly configurable; it hard-codes design
    decisions found to work well in the previous Java-based LensKit code.

    .. note::

        This component must be used with queries containing the user's history,
        either directly in the input or by wiring its query input to the output of a
        user history component (e.g., :class:`~lenskit.basic.UserTrainingHistoryLookup`).

    Stability:
        Caller
    """

    config: UserKNNConfig

    users_: Vocabulary
    "The index of user IDs."
    items_: Vocabulary
    "The index of item IDs."
    user_means_: torch.Tensor | None
    "Mean rating for each known user."
    user_vectors_: torch.Tensor
    "Normalized rating matrix (CSR) to find neighbors at prediction time."
    user_ratings_: csc_array
    "Centered but un-normalized rating matrix (COO) to find neighbor ratings."

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        """
        "Train" a user-user CF model.  This memorizes the rating data in a format that is usable
        for future computations.

        Args:
            ratings(pandas.DataFrame): (user, item, rating) data for collaborative filtering.
        """
        if hasattr(self, "user_ratings_") and not options.retrain:
            return

        ensure_parallel_init()
        rmat = data.interaction_matrix(
            format="torch", field="rating" if self.config.explicit else None
        )
        assert rmat.is_sparse_csr

        if self.config.explicit:
            rmat, means = normalize_sparse_rows(rmat, "center")
            if np.allclose(rmat.values(), 0.0):
                _log.warning("normalized ratings are zero, centering is not recommended")
                warnings.warn(
                    "Ratings seem to have the same value, centering is not recommended.",
                    DataWarning,
                )
        else:
            means = None

        normed, _norms = normalize_sparse_rows(rmat, "unit")

        self.user_vectors_ = normed
        self.user_ratings_ = torch_sparse_to_scipy(rmat).tocsc()
        self.users_ = data.users.copy()
        self.user_means_ = means
        self.items_ = data.items.copy()

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
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
        query = RecQuery.create(query)
        watch = util.Stopwatch()
        log = _log.bind(user_id=query.user_id, n_items=len(items))
        if len(items) == 0:
            log.debug("no candidate items, skipping")
            return ItemList(items, scores=np.nan)

        udata = self._get_user_data(query)
        if udata is None:
            log.debug("user has no ratings, skipping")
            return ItemList(items, scores=np.nan)

        uidx, ratings, umean = udata
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

        nbr_mask = nbr_sims >= self.config.min_sim

        kn_sims = nbr_sims[nbr_mask]
        kn_idxs = nbr_idxs[nbr_mask]
        if len(kn_sims) > 0:
            log.debug(
                "found %d candidate neighbors (of %d total), max sim %0.4f",
                len(kn_sims),
                len(self.users_),
                torch.max(kn_sims).item(),
            )
        else:
            log.warning("no candidate neighbors found")
            return ItemList(items, scores=np.nan)

        assert not torch.any(torch.isnan(kn_sims))

        iidxs = items.numbers(vocabulary=self.items_, missing="negative")
        iidxs = torch.from_numpy(iidxs).to(torch.int64)

        ki_mask = iidxs >= 0
        usable_iidxs = iidxs[ki_mask]

        scores = score_items_with_neighbors(
            log,
            usable_iidxs,
            kn_idxs,
            kn_sims,
            self.user_ratings_,
            self.config.k,
            self.config.min_nbrs,
            self.config.explicit,
        )

        scores += umean

        results = pd.Series(scores, index=items.ids()[ki_mask.numpy()], name="prediction")
        results = results.reindex(items.ids())

        log.debug(
            "scored %d items in %s",
            results.notna().sum(),
            watch,
        )
        return ItemList(items, scores=results.values)  # type: ignore

    def _get_user_data(self, query: RecQuery) -> Optional[UserRatings]:
        "Get a user's data for user-user CF"

        index = self.users_.number(query.user_id, missing=None)

        if query.user_items is None:
            if index is None:
                _log.warning("user %s has no ratings and none provided", query.user_id)
                return None

            assert index >= 0
            row = self.user_vectors_[index].to_dense()
            if self.config.explicit:
                assert self.user_means_ is not None
                umean = self.user_means_[index].item()
            else:
                umean = 0
            return UserRatings(index, row, umean)
        else:
            _log.debug("using provided item history")
            ratings = torch.zeros(len(self.items_), dtype=torch.float32)
            ui_nos = query.user_items.numbers("torch", missing="negative", vocabulary=self.items_)
            ui_mask = ui_nos >= 0

            if self.config.explicit:
                urv = query.user_items.field("rating", "torch")
                if urv is None:
                    _log.warning("user %s has items but no ratings", query.user_id)
                    return None

                umean = urv.mean().item()
                ratings[ui_nos[ui_mask]] = urv[ui_mask] - umean
            else:
                umean = 0
                ratings[ui_nos[ui_mask]] = 1.0

            return UserRatings(index, ratings, umean)


class UserRatings(NamedTuple):
    """
    Dense user ratings.
    """

    index: int | None
    ratings: torch.Tensor
    mean: float


def score_items_with_neighbors(
    log: structlog.stdlib.BoundLogger,
    items: torch.Tensor,
    nbr_rows: torch.Tensor,
    nbr_sims: torch.Tensor,
    ratings: csc_array,
    max_nbrs: int,
    min_nbrs: int,
    average: bool,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    # select a sub-matrix for further manipulation
    items = items.numpy()
    (ni,) = items.shape
    (nrow, ncol) = ratings.shape

    # sort neighbors by similarity
    nbr_order = np.argsort(-nbr_sims)
    nbr_rows = nbr_rows[nbr_order].numpy()
    nbr_sims = nbr_sims[nbr_order].numpy()

    # get the rating rows for our neighbors
    nbr_rates = ratings[nbr_rows, :]

    # which items are scorable?
    counts = np.diff(nbr_rates.indptr)
    min_nbr_mask = counts >= min_nbrs
    is_nbr_mask = min_nbr_mask[items]
    is_scorable = items[is_nbr_mask]

    # get the ratings for requested scorable items
    nbr_rates = nbr_rates[:, is_scorable]
    assert isinstance(nbr_rates, csc_array)
    nbr_rates.sort_indices()
    counts = counts[is_scorable]

    log.debug(
        "scoring items",
        max_count=np.max(counts) if len(counts) else 0,
        nbr_shape=nbr_rates.shape,
    )

    # Now, for our next trick - we have a CSC matrix, whose rows (users) are
    # sorted by decreasing similarity.  So we can *zero* any entries past the
    # first max_neighbors in a row.  This can be done with a little bit of
    # jiggery-pokery.

    # step 1: create a list of column start indices
    starts = np.repeat(nbr_rates.indptr[:-1], counts)
    # step 2: create a ranking from start to end
    ranks = np.arange(nbr_rates.nnz, dtype=np.int32)
    # step 3: subtract the column starts — this will give us numbers within rows
    ranks -= starts
    rmask = ranks >= max_nbrs
    # step 4: zero out rating values for everything past max_nbrs
    nbr_rates.data[rmask] = 0

    # now we can just do a matrix-vector multiply to compute the scores
    results = np.full(ni, np.nan)
    results[is_nbr_mask] = nbr_rates.T @ nbr_sims

    if average:
        nbr_ones = csc_array(
            (np.where(rmask, 0, 1), nbr_rates.indices, nbr_rates.indptr), nbr_rates.shape
        )
        tot_sims = nbr_ones.T @ nbr_sims
        assert np.all(np.isfinite(tot_sims))
        results[is_nbr_mask] /= tot_sims

    return results
