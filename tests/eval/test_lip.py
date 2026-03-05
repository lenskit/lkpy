# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

from lenskit.data import ItemList
from lenskit.metrics.reranking import least_item_promoted


def test_lip_empty_reference():
    ref = ItemList()
    rerank = ItemList([1, 2, 3], ordered=True)
    assert np.isnan(least_item_promoted(ref, rerank))


def test_lip_no_overlap():
    ref = ItemList([10, 11, 12], ordered=True)
    rerank = ItemList([1, 2, 3], ordered=True)
    # none of the reranked items exist in reference
    # lip_rank stays at k, so result = 0
    assert least_item_promoted(ref, rerank, n=3) == 0


def test_lip_perfect_alignment():
    ref = ItemList([1, 2, 3, 4, 5], ordered=True)
    rerank = ItemList([1, 2, 3, 4, 5], ordered=True)
    # top-k reranked items are already top-k in reference
    assert least_item_promoted(ref, rerank, n=3) == 0


def test_lip_promoted_from_below_k():
    ref = ItemList([1, 2, 3, 4, 5, 6], ordered=True)
    rerank = ItemList([6, 2, 3, 1], ordered=True)
    # item 6 is at position 5 in reference but appears in reranked top-3
    # lip_rank = 5, so return = 5 - 3 = 2
    assert least_item_promoted(ref, rerank, n=3) == 2


def test_lip_multiple_promotions():
    ref = ItemList([1, 2, 3, 4, 5, 6, 7], ordered=True)
    rerank = ItemList([7, 6, 2, 3], ordered=True)
    # item 7 is at rank 6, promoted into top-3
    # lip_rank = 6, so result = 6 - 3 = 3
    assert least_item_promoted(ref, rerank, n=3) == 3
