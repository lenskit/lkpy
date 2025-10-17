# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Sparse LInear Methods for Recommendation :cite:p:`ningSLIMSparseLinear2011`.
"""

import warnings

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, PositiveFloat, PositiveInt
from scipy.sparse import csr_array

from lenskit._accel import slim as _slim_accel
from lenskit.data import Dataset, ItemList, RecQuery, Vocabulary
from lenskit.data.matrix import SparseRowArray
from lenskit.diagnostics import DataWarning
from lenskit.logging import get_logger, item_progress
from lenskit.parallel.config import ensure_parallel_init
from lenskit.pipeline.components import Component
from lenskit.training import Trainable, TrainingOptions

_log = get_logger(__name__)


class SLIMConfig(BaseModel):
    l1_reg: PositiveFloat = 0.005
    """
    L₁ regularization strength for SLIM.
    """
    l2_reg: PositiveFloat = 0.01
    """
    L₂ regularization strength for SLIM.
    """
    max_iters: PositiveInt = 50
    """
    Maximum iterations per column.
    """


class SLIMScorer(Component, Trainable):
    """
    Item scorer using Sparse LInear Methods (SLIM).  SLIM was described for
    recommendation by :cite:t`ningSLIMSparseLinear2011`.  This implementation
    closely follows the paper, with some reference to `libslim`_ for
    computational details.  It uses coodrinate descent with soft thresholding
    to estimate the SLIM weight matrix.

    .. _libslim: https://github.com/KarypisLab/SLIM/tree/master/src/libslim
    """

    config: SLIMConfig

    weights: csr_array
    items: Vocabulary

    def train(self, data: Dataset, options: TrainingOptions):
        if hasattr(self, "weights") and not options.retrain:
            return

        ensure_parallel_init()
        ui_matrix = data.interactions().matrix().csr_structure(format="arrow")
        _log.info(
            "training SLIM model with %d interactions for %d items",
            ui_matrix.nnz,
            ui_matrix.dimension,
        )
        iu_matrix = ui_matrix.transpose()

        with item_progress("SLIM vectors", ui_matrix.dimension) as pb:
            weights = _slim_accel.train_slim(
                ui_matrix,
                iu_matrix,
                self.config.l1_reg,
                self.config.l2_reg,
                self.config.max_iters,
                pb,
            )
        weights = pa.chunked_array(weights).combine_chunks()
        weights = SparseRowArray.from_array(weights)
        _log.info("learned %d SLIM weights", weights.nnz)
        self.weights = weights = weights.to_scipy().T.tocsr()
        self.items = data.items

    def __call__(self, query: RecQuery, items: ItemList) -> ItemList:
        u_items = query.user_items
        if u_items is None:
            warnings.warn("no user history available", DataWarning)
            return ItemList(items, scores=np.nan)

        if len(u_items) == 0:
            _log.debug("user %s has no history", query.user_id)
            return ItemList(items, scores=np.nan)

        # get user item numbers
        u_inos = u_items.numbers(vocabulary=self.items, missing="negative")
        u_inos = u_inos[u_inos >= 0]

        # prepare our initial matrix
        x = np.zeros(len(self.items))
        x[u_inos] = 1

        # compute the scores
        all_scores = x @ self.weights

        # finalize result
        scores = np.full(len(items), np.nan, np.float32)
        inos = items.numbers(vocabulary=self.items)
        mask = inos >= 0
        scores[mask] = all_scores[inos[mask]]

        return ItemList(items, scores=scores)
