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
import torch
from pydantic import AliasChoices, BaseModel, Field, PositiveFloat, PositiveInt, field_validator
from scipy.sparse import coo_array, csr_array, sparray
from typing_extensions import Optional, override

from lenskit import _accel
from lenskit.data import Dataset, FeedbackType, ItemList, QueryInput, RecQuery, Vocabulary
from lenskit.data.matrix import sparse_to_arrow
from lenskit.diagnostics import DataWarning
from lenskit.logging import Stopwatch, get_logger, trace
from lenskit.logging.progress import pbh_update
from lenskit.logging.resource import cur_memory, max_memory
from lenskit.parallel import ensure_parallel_init
from lenskit.pipeline import Component
from lenskit.torch import inference_mode, safe_tensor
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

    items_: Vocabulary
    "Vocabulary of item IDs."
    item_means_: np.ndarray[int, np.dtype[np.float32]] | None
    "Mean rating for each known item."
    item_counts_: np.ndarray[int, np.dtype[np.int32]]
    "Number of saved neighbors for each item."
    sim_matrix_: csr_array
    "Similarity matrix (sparse CSR tensor)."
    users_: Vocabulary
    "Vocabulary of user IDs."

    @override
    @inference_mode
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        """
        Train a model.

        The model-training process depends on ``save_nbrs`` and ``min_sim``, but *not* on other
        algorithm parameters.

        Args:
            ratings:
                (user,item,rating) data for computing item similarities.
        """
        if hasattr(self, "items_") and not options.retrain:
            return

        ensure_parallel_init()
        log = _log.bind(n_items=data.item_count, feedback=self.config.feedback)
        # Training proceeds in 2 steps:
        # 1. Normalize item vectors to be mean-centered and unit-normalized
        # 2. Compute similarities with pairwise dot products
        timer = Stopwatch()
        log.info("begining IKNN training")

        field = "rating" if self.config.explicit else None
        rmat = data.interactions().matrix().scipy(field, layout="coo")
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
        ui_mat = sparse_to_arrow(rmat.tocsr())
        iu_mat = sparse_to_arrow(rmat.T.tocsr())
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
        assert isinstance(smat, pa.LargeListArray)
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
        assert smat.offsets[-1].as_py() == len(
            smat.values
        ), f"{smat.offsets[-1]} != {len(smat.values)}"

        self.items_ = data.items
        self.users_ = data.users
        self.item_means_ = np.asarray(means)
        self.item_counts_ = np.diff(smat.offsets.to_numpy())
        self.sim_matrix_ = smat
        log.debug("[%s] done, memory use %s", timer, max_memory())

    def _center_ratings(self, log, timer, rmat: coo_array) -> tuple[sparray, np.ndarray | None]:
        if self.config.explicit:
            rmat = rmat.tocsc()
            counts = np.diff(rmat.indptr)
            sums = rmat.sum(axis=0)
            means = sums / counts
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
        cmat = rmat / norms
        assert cmat.shape == rmat.shape
        log.debug("[%s] normalized, memory use %s", timer, max_memory())
        return cmat

    @override
    @inference_mode
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
        ri_nums = ratings.numbers(format="numpy", vocabulary=self.items_, missing="negative")
        ri_mask = ri_nums >= 0
        ri_valid_nums = ri_nums[ri_mask]
        n_valid = len(ri_valid_nums)
        trace(log, "%d of %d rated items in model", n_valid, len(ratings))

        if self.config.explicit:
            ri_vals = ratings.field("rating", "numpy")
            if ri_vals is None:
                raise RuntimeError("explicit-feedback scorer must have ratings")
            ri_vals = np.require(ri_vals[ri_mask], np.float32)
        else:
            ri_vals = np.full(n_valid, 1.0, dtype=np.float32)

        # mean-center the rating array
        if self.item_means_ is not None:
            ri_vals -= self.item_means_[ri_valid_nums]

        # convert target item information
        ti_nums = items.numbers(vocabulary=self.items_, missing="negative")
        ti_mask = ti_nums >= 0
        ti_valid_nums = ti_nums[ti_mask]
        trace(log, "attempting to score %d of %d items", len(ti_valid_nums), len(items))

        # subset the model to rated and target items
        model = self.sim_matrix_
        trace(log, "subsetting matrix rows", n_rates=len(ri_valid_nums), shape=model.shape)
        model = model[ri_valid_nums, :]
        assert isinstance(model, csr_array)
        trace(log, "subsetting matrix columns", n_targets=len(ti_valid_nums), shape=model.shape)
        model = model[:, ti_valid_nums]
        assert isinstance(model, csr_array)

        # which neighborhoods are usable? (at least min neighbors)
        m_ind = csr_array(
            (np.ones(model.nnz, np.int32), model.indices, model.indptr), shape=model.shape
        )
        sizes = m_ind.sum(0)
        del m_ind
        assert isinstance(sizes, np.ndarray) and len(sizes) == model.shape[1]
        scorable = sizes >= self.config.min_nbrs

        # fast-path neighborhoods that fit within max neighbors
        fast = sizes <= self.config.max_nbrs
        ti_fast_mask = ti_mask.copy()
        ti_fast_mask[ti_mask] = scorable & fast

        scores = np.full(len(items), np.nan, dtype=np.float32)
        fast_mod = model[:, scorable & fast]
        if self.config.explicit:
            scores[ti_fast_mask] = ri_vals @ fast_mod
            scores[ti_fast_mask] /= fast_mod.sum(axis=0)
        else:
            scores[ti_fast_mask] = fast_mod.sum(axis=0)

        # slow path: neighborhoods that we need to truncate. we will convert to
        # PyTorch, make a dense matrix (this is usually small enough to be
        # usable), and use the Torch topk function.
        slow_mat = model.T[~fast, :]
        # assert isinstance(slow_mat, csr_array)
        n_slow, _ = slow_mat.shape
        if n_slow:
            # mask for the slow items.
            ti_slow_mask = ti_mask.copy()
            ti_slow_mask[ti_mask] = ~fast

            slow_mat = safe_tensor(slow_mat.toarray())
            slow_trimmed, slow_inds = torch.topk(slow_mat, self.config.max_nbrs)
            assert slow_trimmed.shape == (n_slow, self.config.max_nbrs)
            if self.config.explicit:
                svals = safe_tensor(ri_vals)[slow_inds]
                assert svals.shape == slow_trimmed.shape
                scores[ti_slow_mask] = torch.sum(slow_trimmed * svals, axis=1).numpy()
                scores[ti_slow_mask] /= torch.sum(slow_trimmed, axis=1).numpy()
            else:
                scores[ti_slow_mask] = torch.sum(slow_trimmed, axis=1).numpy()

        # re-add the mean ratings in implicit feedback
        if self.item_means_ is not None:
            scores[ti_mask] += self.item_means_[ti_valid_nums]

        log.debug(
            "scored %d items",
            int(np.isfinite(scores).sum()),
        )

        return ItemList(items, scores=scores)


@torch.jit.script
def _sim_row(
    item: int, matrix: torch.Tensor, row: torch.Tensor, min_sim: float, max_nbrs: Optional[int]
) -> tuple[int, torch.Tensor, torch.Tensor]:
    nitems, nusers = matrix.shape
    if len(row.indices()) == 0:
        return 0, torch.zeros((0,), dtype=torch.int32), torch.zeros((0,), dtype=torch.float64)

    # _item_dbg(item, f"comparing item with {row.indices().shape[1]} users")
    # _item_dbg(item, f"row norm {torch.linalg.vector_norm(row.values()).item()}")
    row = row.to_dense()
    sim = torch.mv(matrix, row.to(torch.float64))
    sim[item] = 0

    mask = sim >= min_sim
    cols = torch.nonzero(mask)[:, 0].to(torch.int32)
    vals = sim[mask]
    # _item_dbg(item, f"found {len(vals)} acceptable similarities")
    assert len(cols) == torch.sum(mask)

    if max_nbrs is not None and max_nbrs > 0 and max_nbrs < vals.shape[0]:
        # _item_dbg(item, "truncating similarities")
        vals, cis = torch.topk(vals, max_nbrs, sorted=False)
        cols = cols[cis]
        order = torch.argsort(cols)
        cols = cols[order]
        vals = vals[order]

    return len(cols), cols, torch.clamp(vals, -1, 1)


@torch.jit.script
def _sim_block(
    matrix: torch.Tensor, start: int, end: int, min_sim: float, max_nbrs: Optional[int], pbh: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    "Compute a single block of the similarity matrix"
    bsize = end - start

    counts = torch.zeros(bsize, dtype=torch.int32)
    columns = []
    values = []

    for i in range(start, end):
        c, cs, vs = _sim_row(i, matrix, matrix[i], min_sim, max_nbrs)
        counts[i - start] = c
        columns.append(cs)
        values.append(vs)
        pbh_update(pbh, 1)

    return counts, torch.cat(columns), torch.cat(values).to(torch.float32)


@torch.jit.script
def _sim_blocks(
    matrix: torch.Tensor, min_sim: float, max_nbrs: Optional[int], block_size: int, pbh: str
) -> torch.Tensor:
    "Compute the similarity matrix with blocked matrix-matrix multiplies"
    nitems, nusers = matrix.shape

    jobs: list[torch.jit.Future[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []

    for start in range(0, nitems, block_size):
        end = min(start + block_size, nitems)
        jobs.append(torch.jit.fork(_sim_block, matrix, start, end, min_sim, max_nbrs, pbh))  # type: ignore

    counts = [torch.tensor([0], dtype=torch.int32)]
    columns = []
    values = []

    for job in jobs:
        cts, cis, vs = job.wait()
        counts.append(cts)
        columns.append(cis)
        values.append(vs)

    c_cat = torch.cat(counts)
    crow_indices = torch.cumsum(c_cat, 0, dtype=torch.int32)
    assert len(crow_indices) == nitems + 1
    col_indices = torch.cat(columns)
    c_values = torch.cat(values)
    assert crow_indices[nitems] == len(col_indices)
    assert crow_indices[nitems] == len(c_values)

    return torch.sparse_csr_tensor(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=c_values,
        size=(nitems, nitems),
    )
