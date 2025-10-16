# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from lenskit.data import ItemList
from lenskit.metrics import GeometricRankWeight, RankWeight


def entropy(
    items: ItemList,
    *,
    categories: str,
    k: int | None = None,
) -> float:
    """
    Compute the Shannon entropy over categorical distributions in an ItemList.

    This metric measures the diversity of categories represented in the ItemList.
    Higher entropy indicates more diverse category representation.

    Args:
        items:
            The ItemList to evaluate.
        categories:
            The name of the field containing category data.
        k:
            The depth to evaluate. If None, uses the full list length.

    Returns:
        The Shannon entropy.

    Stability:
        Experimental
    """
    if k is None:
        k = len(items)

    top_k_items = items[:k]

    category_lists = _extract_categories(top_k_items, categories)
    if category_lists is None:
        return np.nan

    category_counts = defaultdict(int)
    for cats in category_lists:
        if not cats:
            continue
        for cat in cats:
            category_counts[cat] += 1

    if not category_counts:
        return np.nan

    total_count = sum(category_counts.values())
    category_probs = {cat: count / total_count for cat, count in category_counts.items()}

    entropy_value = -sum(p * np.log2(p) for p in category_probs.values() if p > 0)
    return entropy_value


def rank_biased_entropy(
    items: ItemList,
    *,
    categories: str | None,
    weight: RankWeight | None = None,
    k: int | None = None,
) -> float:
    """
    Compute rank-biased entropy (RBE) over categorical distributions in an ItemList.

    This metric measures diversity of categories with position-based weights.
    It gives more importance to items at higher ranks. Higher values indicate more
    diverse category representation at the top of the list.

    Args:
        items:
            The ItemList to evaluate.
        categories:
            The name of the field containing category data.
        weight:
            RankWeight instance to compute item weights. Defaults to GeometricRankWeight(0.85).
        k:
            The depth to evaluate. If None, uses the full list length.

    Returns:
        The rank-biased entropy.

    Stability:
        Experimental
    """
    if k is None:
        k = len(items)

    top_k_items = items[:k]

    if weight is None:
        weight = GeometricRankWeight(0.85)

    category_lists = _extract_categories(top_k_items, categories)
    if category_lists is None:
        return np.nan

    ranks = np.arange(1, k + 1)
    weights = weight.weight(ranks)

    weighted_counts = defaultdict(float)
    for cats, w in zip(category_lists, weights):
        if not cats:
            continue
        if isinstance(cats, str):
            cats = [cats]
        for cat in cats:
            weighted_counts[cat] += float(w)

    if not weighted_counts:
        return np.nan

    total_weight = sum(weighted_counts.values())
    if total_weight == 0:
        return 0.0

    category_probs = [count / total_weight for count in weighted_counts.values()]
    entropy_value = -sum(p * np.log2(p) for p in category_probs if p > 0)
    return entropy_value


def _extract_categories(
    items: ItemList,
    categories: str | None,
) -> Sequence[Sequence[str]] | None:
    """
    Helper function to extract categories from items.
    """
    if categories is None:
        return None

    field_data = items.field(categories)
    if field_data is None:
        return None

    return [
        list(cats) if isinstance(cats, (list, tuple, np.ndarray)) else [cats] for cats in field_data
    ]
