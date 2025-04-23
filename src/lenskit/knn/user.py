# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
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
import pyarrow as pa
import torch
from pydantic import AliasChoices, BaseModel, Field, PositiveFloat, PositiveInt, field_validator
from scipy.sparse import csr_array
from typing_extensions import NamedTuple, Optional, override

from lenskit._accel import knn
from lenskit.data import Dataset, FeedbackType, ItemList, QueryInput, RecQuery
from lenskit.data.matrix import SparseRowArray
from lenskit.data.vocab import Vocabulary
from lenskit.diagnostics import DataWarning
from lenskit.logging import Stopwatch, get_logger
from lenskit.math.sparse import normalize_sparse_rows, torch_sparse_to_scipy
from lenskit.parallel.config import ensure_parallel_init
from lenskit.pipeline import Component
from lenskit.torch import inference_mode, sparse_row
from lenskit.training import Trainable, TrainingOptions

_log = get_logger(__name__)


class UserKNNConfig(BaseModel, extra="forbid"):
    "Configuration for :class:`ItemKNNScorer`."

    max_nbrs: PositiveInt = Field(20, validation_alias=AliasChoices("max_nbrs", "nnbrs", "k"))
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
    user_ratings_: SparseRowArray
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
        ).to(torch.float32)
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
        normed = normed.to(torch.float32)

        self.user_vectors_ = normed.detach()
        rmat = torch_sparse_to_scipy(rmat)
        assert isinstance(rmat, csr_array)
        self.user_ratings_ = SparseRowArray.from_scipy(rmat, values=self.config.explicit)
        self.users_ = data.users
        self.user_means_ = means
        self.items_ = data.items

    @override
    @inference_mode
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
        watch = Stopwatch()
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
        nbr_sims = torch.mv(self.user_vectors_, ratings)
        assert nbr_sims.shape == (len(self.users_),)
        if uidx is not None:
            # zero out the self-similarity
            nbr_sims[uidx] = 0

        # get indices for these neighbors
        nbr_idxs = np.arange(len(self.users_), dtype=np.int32)

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
            log.debug("no candidate neighbors found, cannot score")
            return ItemList(items, scores=np.nan)

        assert not torch.any(torch.isnan(kn_sims))

        iidxs = items.numbers(format="torch", vocabulary=self.items_, missing="negative").to(
            torch.int64
        )

        ki_mask = iidxs >= 0
        usable_iidxs = iidxs[ki_mask]

        usable_iidxs = pa.array(usable_iidxs, pa.int32())
        kn_idxs = pa.array(kn_idxs, pa.int32())
        kn_sims = pa.array(kn_sims.numpy(), pa.float32())

        if self.config.explicit:
            scores = knn.user_score_items_explicit(
                usable_iidxs,
                kn_idxs,
                kn_sims,
                self.user_ratings_,
                self.config.max_nbrs,
                self.config.min_nbrs,
            )
        else:
            scores = knn.user_score_items_implicit(
                usable_iidxs,
                kn_idxs,
                kn_sims,
                self.user_ratings_,
                self.config.max_nbrs,
                self.config.min_nbrs,
            )

        scores = scores.to_numpy(zero_copy_only=False, writable=True)
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

            index = int(index)
            row = sparse_row(self.user_vectors_, index)
            row = row.to_dense()
            if self.config.explicit:
                assert self.user_means_ is not None
                umean = self.user_means_[index].item()
            else:
                umean = 0
            return UserRatings(index, row, umean)
        elif len(query.user_items) == 0:
            return None
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

                urv = urv.to(torch.float32)
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
