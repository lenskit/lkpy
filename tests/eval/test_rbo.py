# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pytest import approx

from lenskit.data import ItemList
from lenskit.metrics import call_metric
from lenskit.metrics.reranking import rank_biased_overlap


def test_rbo_empty_reference():
    ref = ItemList()
    rerank = ItemList([1, 2, 3], ordered=True)
    assert call_metric(rank_biased_overlap, ref, rerank) == 0.0


def test_rbo_empty_rerank():
    ref = ItemList([1, 2, 3], ordered=True)
    rerank = ItemList()
    assert call_metric(rank_biased_overlap, ref, rerank) == 0.0


def test_rbo_perfect_match():
    ref = ItemList([1, 2, 3, 4, 5], ordered=True)
    rerank = ItemList([1, 2, 3, 4, 5], ordered=True)
    score = call_metric(rank_biased_overlap, ref, rerank, k=5, p=0.9)
    assert score > 0.89
    assert score <= 1.0


def test_rbo_no_overlap():
    ref = ItemList([1, 2, 3], ordered=True)
    rerank = ItemList([4, 5, 6], ordered=True)
    assert call_metric(rank_biased_overlap, ref, rerank, k=3, p=0.9) == approx(0.0)


def test_rbo_partial_overlap():
    ref = ItemList([1, 2, 3, 4, 5], ordered=True)
    rerank = ItemList([3, 2, 6, 7, 1], ordered=True)
    score = call_metric(rank_biased_overlap, ref, rerank, k=5, p=0.9)
    # RBO should be between 0 and 1
    assert 0 < score < 1


def test_rbo_k_smaller_than_list():
    ref = ItemList([1, 2, 3, 4, 5], ordered=True)
    rerank = ItemList([5, 4, 3, 2, 1], ordered=True)
    # consider only top 3
    score = call_metric(rank_biased_overlap, ref, rerank, k=3, p=0.9)
    assert 0 <= score <= 1
