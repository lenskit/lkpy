# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

import pytest

from lenskit.data import ItemList
from lenskit.metrics import entropy, rank_biased_entropy


def test_entropy_empty_list():
    recs = ItemList(item_ids=[1, 2], ordered=True)
    result = entropy(recs, categories="topic")
    assert np.isnan(result)


def test_entropy_single_category():
    recs = ItemList(
        item_ids=[1, 2, 3],
        topic=["sports", "sports", "sports"],
        ordered=True,
    )
    result = entropy(recs, categories="topic")
    assert result == pytest.approx(0.0)


def test_entropy_diverse_categories():
    recs = ItemList(
        item_ids=[1, 2, 3],
        topic=["health", "sports", "tech"],
        ordered=True,
    )
    result = entropy(recs, categories="topic", k=3)
    assert result == pytest.approx(1.584962500721156)


def test_entropy_k_subset():
    recs = ItemList(
        item_ids=[1, 2, 3, 4],
        topic=["health", "sports", "tech", "health"],
        ordered=True,
    )

    score_full = entropy(recs, categories="topic")
    score_top2 = entropy(recs, categories="topic", k=2)

    assert score_full >= score_top2


def test_entropy_items_with_empty_category():
    recs = ItemList(
        item_ids=[1, 2, 3],
        topic=["", "sports", ""],
        ordered=True,
    )
    result = entropy(recs, categories="topic")
    assert result == pytest.approx(0.9182958340544896)
