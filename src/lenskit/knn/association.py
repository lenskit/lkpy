# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Association-rule (conditional probability and lift) nearest-neighbor
recommendation.
"""

from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, NonNegativeFloat, PositiveInt
from scipy.sparse import csr_array

from lenskit.data import Dataset, ItemList, RecQuery, Vocabulary
from lenskit.logging import Stopwatch, get_logger
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_log = get_logger(__name__)

type AssociationMethod = Literal["probability", "lift"]
"""
Methods used to compute association scores.
"""


class AssociationConfig(BaseModel, extra="forbid"):
    """
    Configuration options for :class:`AssociationScorer`.
    """

    method: AssociationMethod = "probability"
    """
    The formula to use for computing item association level.
    """

    damping: NonNegativeFloat = 0.0
    r"""
    Damping factor (:math:`\kappa`) for `biased lift`_.

    .. _biased lift: https://md.ekstrandom.net/blog/2025/01/biased-lift
    """

    max_nbrs: PositiveInt | None = None
    """
    Maximum number of reference items used to score each target item.  If
    ``None``, items are scored by the mean of their score with respect to each
    reference item.  If a positive integer :math:`n`, then only the :math:`n`
    most-related reference items are used.  :math:`n=1` is equivalent to using
    only the maximum relatedness.
    """


class AssociationScorer(Component[ItemList], Trainable):
    r"""
    Item scorer using association rules to compute item relatedness.

    This scorer can compute item associations with three formulas:

    - Conditional probability (:math:`P[i|j]`), by setting
      :attr:`~AssociationConfig.method` to ``"probability"``.
    - Lift (:math:`\frac{P[i,j]}{P[i]P[j]}`), by setting
      :attr:`~AssociationConfig.method` to ``"lift"`` and
      :attr:`~AssociationConfig.damping` to 0.
    - `Biased lift`), by setting :attr:`~AssociationConfig.method` to ``"lift"``
      and :attr:`~AssociationConfig.damping` (:math:`\kappa`) to a positive
      value.

    An empty query will recommend no items.

    .. _Biased lift: https://md.ekstrandom.net/blog/2025/01/biased-lift
    """

    config: AssociationConfig

    items: Vocabulary
    assoc_scores: csr_array
    """
    Sparse matrix of item association scores, with reference items on rows and
    target items on columns.
    """

    def is_trained(self):
        return hasattr(self, "assoc_scores")

    def train(self, data: Dataset, options: TrainingOptions):
        timer = Stopwatch()

        # the core computation is to extract a *co-occurrance* matrix, which we will
        # then adjust with marginal probabilities to obtain lift.
        interact = data.interactions()

        # we can't just use user_count, because we might have sessions
        n_groups = interact.matrix().n_rows

        # RelationshipSet now has a method to compute co-occurrences!
        # NOTE: this is very slow on the MovieLens data
        cooc = interact.co_occurrences("item")
        _log.info("[%s] computed %d co-occurrences", timer, cooc.nnz)
        # make a copy of the data so we can modify it and so it is floating-point
        # cooc is a COO sparse matrix, with 3 component arrays:
        # - data, the actual values
        # - row, the row numbers of each entry
        # - col, the column numbers of each entry
        cooc_vals = cooc.data.astype(np.float32)

        # Let's also get the marginal probabilities for each item by dividing count by total
        item_counts = cast(NDArray[np.int32], data.item_stats()["count"].values)

        # Now, to compute lift: our matrix is a *coo* matrix, so it has
        # row and column attributes. We can use these to look up the
        # unconditional probabilities from our item probability array.
        _log.info("[%s] converting to lift", timer)
        # divide by row probs to get conditional probabilities
        cooc_vals /= item_counts[cooc.row] + self.config.damping
        if self.config.method == "lift":
            cooc_vals *= n_groups
            cooc_vals /= item_counts[cooc.col] + self.config.damping
            # now we have (possibly damped) lift!

        # now we can save the results, and we're done
        _log.info("[%s] training finished, saving %d results", timer, cooc.nnz)
        self.items = data.items
        self.item_freqs = item_counts
        self.assoc_scores = csr_array((cooc_vals, (cooc.row, cooc.col)), shape=cooc.shape)

    def __call__(self, query: RecQuery, items: ItemList):
        log = _log
        if query.user_id is not None:
            log = log.bind(user_id=query.user_id)
        if query.query_id is not None:
            log = log.bind(query_id=query.query_id)

        ref_items = []
        if query.query_items:
            ref_items = query.query_items.numbers(vocabulary=self.items, missing="negative")
            ref_items = ref_items[ref_items >= 0]

        if len(ref_items) == 0:
            log.debug("no reference items, skipping", user=query.user_id)
            return ItemList(items, scores=np.nan)

        log.debug("scoring with %d reference items", len(ref_items))
        assoc_mat = self.assoc_scores[ref_items, :].todense()
        if self.config.max_nbrs == 1:
            scores = np.max(assoc_mat, axis=0)
        elif self.config.max_nbrs is None:
            scores = np.mean(assoc_mat, axis=0)
        else:
            raise NotImplementedError("limited reference items not yet implemented")

        assert scores.shape == (len(self.items),)

        tgt_items = items.numbers(vocabulary=self.items, missing="negative")
        tgt_items = tgt_items[tgt_items >= 0]

        tgt_scores = ItemList(item_nums=tgt_items, scores=scores[tgt_items], vocabulary=self.items)
        return items.update(tgt_scores)
