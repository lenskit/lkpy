# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
EASE scoring model.
"""

import numpy as np
import scipy.linalg as spla
from humanize import metric, naturalsize
from pydantic import BaseModel, PositiveFloat

from lenskit.data import Dataset, ItemList, RecQuery, Vocabulary
from lenskit.data.matrix import COOStructure, fast_col_cooc
from lenskit.data.types import NPMatrix
from lenskit.logging import Stopwatch, get_logger, item_progress
from lenskit.parallel import ensure_parallel_init
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_log = get_logger(__name__)


class EASEConfig(BaseModel):
    """
    Configuration for :class:`EASEScorer`.
    """

    regularization: PositiveFloat = 0.1
    """
    Regularization term for EASE.
    """


class EASEScorer(Component[ItemList], Trainable):
    """
    Score items using EASE :citep:`steckEmbarrassinglyShallowAutoencoders2019`.
    """

    config: EASEConfig
    """
    EASE configuration.
    """

    items: Vocabulary
    """
    Items known at training time.
    """

    weights: NPMatrix[np.float32]
    """
    Item interpolation weight matrix.
    """

    def train(self, data: Dataset, options: TrainingOptions | None = None):
        if options is None:
            options = TrainingOptions()

        if hasattr(self, "weights") and not options.retrain:
            return

        ensure_parallel_init()

        n_items = data.item_count
        n_users = data.user_count
        rates = data.interactions().matrix(row_entity="item", col_entity="user")
        log = _log.bind(n_items=n_items)
        log.info("training EASE model")

        # trim out single-user items
        log.debug("finding recommendable items")
        rate_csr = rates.csr_structure()
        item_nnz = rate_csr.row_nnzs
        ok_items = item_nnz > 1
        n_ok = int(np.sum(ok_items))
        log.debug("keeping %d items", n_ok)
        iid_map = np.zeros(n_items, dtype=np.int32)
        iid_map[ok_items] = np.arange(n_ok, dtype=np.int32)

        log.debug("filtering interaction matrix")
        rate_coo = rates.coo_structure()

        kept_ents = ok_items[rate_coo.row_numbers]
        kept_inos = rate_coo.row_numbers[kept_ents]
        kept_unos = rate_coo.col_numbers[kept_ents]
        kept_coo = COOStructure(kept_unos, iid_map[kept_inos], shape=(n_users, n_ok))
        del kept_ents, rate_csr, rate_coo, rates

        log.debug("computing co-occurrance matrix")
        timer = Stopwatch()
        with item_progress("Counting co-occurrances", len(kept_inos)) as pb:
            cooc = fast_col_cooc(kept_coo, progress=pb)

        log.info("computed %s co-occurances in %s", metric(cooc.nnz), timer)

        log.debug("converting and densifying matrix")
        cooc = cooc.astype(np.float32).toarray()
        log.debug("made %s dense matrix", naturalsize(cooc.nbytes))

        log.debug("adding regularization term")
        di = np.diag_indices(n_ok)
        cooc[di] = item_nnz[ok_items] + self.config.regularization

        log.debug("inverting Gram-matrix")
        timer = Stopwatch()
        mat = spla.inv(cooc, assume_a="pos")
        log.info("inverted co-occurrance matrix in %s", timer)

        mat /= np.diag(mat)
        mat[di] = 0

        log.debug("assembling trained item vocabulary")
        items = data.items.id_array()
        items = items.filter(ok_items)
        self.items = Vocabulary(items, name="item", reorder=False)

        log.info("finished training EASE for %d items", len(self.items))
        self.weights = mat

    def __call__(self, query: RecQuery, items: ItemList):
        log = _log.bind(user=query.user_id)

        q_items = query.query_items
        if q_items is None:
            log.debug("no history items, cannot score")
            return ItemList(items, score=np.nan)

        q_inos = q_items.numbers(vocabulary=self.items, missing="negative")
        q_ok = q_inos >= 0
        if not np.any(q_ok):
            log.debug("no usable history items, cannot score")
            return ItemList(items, score=np.nan)
        q_good = q_inos[q_ok]

        t_inos = items.numbers(vocabulary=self.items, missing="negative")
        t_ok = t_inos >= 0
        t_good = t_inos[t_ok]

        N = len(self.items)
        q_vec = np.zeros(N, dtype=np.float32)
        q_vec[q_good] = 1.0

        _log.debug("multiplying matrix for %d items", np.sum(q_ok))
        scores = self.weights @ q_vec
        assert scores.shape == (N,)

        scored = ItemList(item_nums=t_good, scores=scores[t_good], vocabulary=self.items)
        return items.update(scored)
