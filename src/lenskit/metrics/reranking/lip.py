# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from lenskit.data import ItemList


def least_item_promoted(reference: ItemList, reranked: ItemList, k: int = 10) -> float:
    """
    Compute the Least Item Promoted (LIP) metric.

    This metric identifies the item in the top-k reranked list that had the
    highest (worst) rank in the reference, and returns how many positions
    it was promoted from beyond k.

    Args:
        reference:
            The original/base ranking.
        reranked:
            The reranked list.
        k:
            The depth to evaluate (default 10).

    Returns:
        The rank distance of the least-promoted item, or NaN if the base
        ranking is empty.
    """
    if len(reference) == 0:
        return np.nan

    base_ids = reference.ids()
    reranked_ids = reranked[:k].ids()

    lip_rank = k
    for item_id in reranked_ids:
        indices = np.where(base_ids == item_id)[0]
        if indices.size > 0:
            rank = indices[0]
            if rank > lip_rank:
                lip_rank = rank

    return lip_rank - k
