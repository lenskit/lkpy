# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Item-based k-NN collaborative filtering.
"""

# pyright: basic
from __future__ import annotations

import gc
import warnings

import numpy as np
import pyarrow as pa
import scipy.sparse.linalg as spla
from pydantic import AliasChoices, BaseModel, Field, PositiveFloat, PositiveInt, field_validator
from scipy.sparse import coo_array, sparray
from typing_extensions import override

from lenskit import _accel
from lenskit.data import Dataset, FeedbackType, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.data.matrix import SparseRowArray
from lenskit.diagnostics import DataWarning
from lenskit.logging import Stopwatch, get_logger, trace
from lenskit.logging.resource import cur_memory, max_memory
from lenskit.parallel import ensure_parallel_init
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_log = get_logger(__name__)
MAX_BLOCKS = 1024


class ItemKNNConfig(BaseModel, extra="forbid"):
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
    save_nbrs: PositiveInt | None = None
    """
    The number of neighbors to save per item in the trained model (``None`` for
    unlimited).
    """
    feedback: FeedbackType = "explicit"
    """
    The type of input data to use (explicit or implicit).  This affects data
    pre-processing and aggregation.
    """
    block_size: int = 250
    """
    The block size for computing item similarity blocks in parallel.  Only
    affects performance, not behavior.
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


class ItemKNNScorer(Component[ItemList], Trainable):
    """
    Item-item nearest-neighbor collaborative filtering feedback. This item-item
    implementation is based on the description of item-based CF by
    :cite:t:`deshpande:iknn` and hard-codes several design decisions found to
    work well in the previous Java-based LensKit code :cite:p:`lenskit-java`. In
    explicit-feedback mode, its output is equivalent to that of the Java
    version.

    .. note::

        This component must be used with queries containing the user's history,
        either directly in the input or by wiring its query input to the output of a
        user history component (e.g., :class:`~lenskit.basic.UserTrainingHistoryLookup`).

    Stability:
        Caller
    """

    config: ItemKNNConfig

    items: Vocabulary
    "Vocabulary of item IDs."
    item_means: np.ndarray[int, np.dtype[np.float32]] | None
    "Mean rating for each known item."
    item_counts: np.ndarray[int, np.dtype[np.int32]]
    "Number of saved neighbors for each item."
    sim_matrix: SparseRowArray
    "Similarity matrix (sparse CSR tensor)."

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        """
        Train a model.

        The model-training process depends on ``save_nbrs`` and ``min_sim``, but *not* on other
        algorithm parameters.

        Args:
            ratings:
                (user,item,rating) data for computing item similarities.
        """
        if hasattr(self, "sim_matrix") and not options.retrain:
            return

        ensure_parallel_init()
        log = _log.bind(n_items=data.item_count, feedback=self.config.feedback)
        # Training proceeds in 2 steps:
        # 1. Normalize item vectors to be mean-centered and unit-normalized
        # 2. Compute similarities with pairwise dot products
        timer = Stopwatch()
        log.info("begining IKNN training")

        field = "rating" if self.config.explicit else None
        rmat = data.interactions().matrix().scipy(field, layout="coo").astype(np.float32)
        n_users, n_items = rmat.shape
        log.info(
            "[%s] made sparse matrix",
            timer,
            n_ratings=rmat.nnz,
            n_users=data.user_count,
        )

        rmat, means = self._center_ratings(log, timer, rmat)
        rmat = self._normalize_rows(log, timer, rmat)

        # convert matrix & its transpose to Arrow for Rust computation
        ui_mat = SparseRowArray.from_scipy(rmat.tocsr())
        iu_mat = SparseRowArray.from_scipy(rmat.T.tocsr())
        del rmat
        log.debug("[%s] prepared working matrices, memory use %s", timer, max_memory())

        log.info("[%s] computing similarity matrix", timer)
        smat = _accel.knn.compute_similarities(
            ui_mat, iu_mat, (n_users, n_items), self.config.min_sim, self.config.save_nbrs
        )
        log.debug("[%s] computed, memory use %s", timer, max_memory())
        assert isinstance(smat, list)
        smat = pa.chunked_array(smat)
        smat = smat.combine_chunks()
        assert pa.types.is_large_list(smat.type)
        smat = SparseRowArray.from_array(smat)
        gc.collect()
        log.debug(
            "[%s] combined chunks, memory use %s (peak %s)", timer, cur_memory(), max_memory()
        )
        lengths = np.diff(smat.offsets)

        log.info(
            "[%s] found neighborhoods for %d of %d items",
            timer,
            np.sum(lengths > 0),
            n_items,
        )

        log.info("[%s] computed %d neighbor pairs", timer, len(smat.values))
        assert smat.offsets[-1].as_py() == len(smat.values), (
            f"{smat.offsets[-1]} != {len(smat.values)}"
        )

        self.items = data.items
        self.item_means = np.asarray(means)
        self.item_counts = np.diff(smat.offsets.to_numpy())
        self.sim_matrix = smat
        log.debug("[%s] done, memory use %s", timer, max_memory())

    def _center_ratings(self, log, timer, rmat: coo_array) -> tuple[sparray, np.ndarray | None]:
        if self.config.explicit:
            rmat = rmat.tocsc()
            counts = np.diff(rmat.indptr)
            sums = rmat.sum(axis=0)
            means = np.zeros(sums.shape, dtype=np.float32)
            # manual divide to avoid division by zero
            np.divide(sums, counts, out=means, where=counts > 0)
            rmat.data = rmat.data - np.repeat(means, counts)
            if np.allclose(rmat.data, 0.0):
                log.warning("normalized ratings are zero, centering is not recommended")
                warnings.warn(
                    "Ratings seem to have the same value, centering is not recommended.",
                    DataWarning,
                )
            log.debug("[%s] centered, memory use %s", timer, max_memory())
            return rmat, means
        else:
            return rmat, None

    def _normalize_rows(self, log, timer, rmat: sparray) -> coo_array:
        norms = spla.norm(rmat, 2, axis=0)
        # clamp small values to avoid divide by 0 (only appear when an entry is all 0)
        cmat = rmat / np.maximum(norms, np.finfo("f4").smallest_normal)
        assert cmat.shape == rmat.shape
        log.debug("[%s] normalized, memory use %s", timer, max_memory())
        return cmat

    @override
    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)
        log = _log.bind(user_id=query.user_id, n_items=len(items))
        trace(log, "beginning prediction")

        ratings = query.user_items
        if ratings is None or len(ratings) == 0:
            if ratings is None:
                warnings.warn("no user history, did you omit a history component?", DataWarning)
            log.debug("user has no history, returning")
            return ItemList(items, scores=np.nan)

        # set up rating array
        # get rated item positions & limit to in-model items
        ri_nums = ratings.numbers(format="numpy", vocabulary=self.items, missing="negative")
        ri_mask = ri_nums >= 0
        ri_arr = pa.array(ri_nums, mask=~ri_mask)
        n_invalid = ri_arr.null_count
        n_valid = len(ratings) - n_invalid
        trace(log, "%d of %d rated items in model", n_valid, len(ratings))

        # convert target item information
        ti_nums = items.numbers(vocabulary=self.items, missing="negative")
        ti_mask = ti_nums >= 0
        ti_arr = pa.array(ti_nums, mask=~ti_mask)
        trace(log, "attempting to score %d of %d items", len(items) - ti_arr.null_count, len(items))

        if self.config.explicit:
            ri_vals = ratings.field("rating", "numpy")
            if ri_vals is None:
                raise RuntimeError("explicit-feedback scorer must have ratings")
            ri_vals = ri_vals.astype(np.float32, copy=True)

            # mean-center the rating array
            assert self.item_means is not None
            ri_vals[ri_mask] -= self.item_means[ri_nums[ri_mask]]
            ri_vals = pa.array(ri_vals, mask=~ri_mask)

            scores = _accel.knn.score_explicit(
                self.sim_matrix,
                ri_arr,
                ri_vals,
                ti_arr,
                self.config.max_nbrs,
                self.config.min_nbrs,
            ).to_numpy(zero_copy_only=False, writable=True)
            scores[ti_mask] += self.item_means[ti_nums[ti_mask]]

        else:
            scores = _accel.knn.score_implicit(
                self.sim_matrix, ri_arr, ti_arr, self.config.max_nbrs, self.config.min_nbrs
            ).to_numpy(zero_copy_only=False)

        log.debug(
            "scored %d items",
            int(np.isfinite(scores).sum()),
        )

        return ItemList(items, scores=scores)
