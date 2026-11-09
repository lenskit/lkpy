# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pyarrow as pa
import torch

import pytest

from lenskit.data import ItemList, Vocabulary


@pytest.mark.parametrize("n", [None, -1, 2, 0])
@pytest.mark.parametrize("score_source", [None, "rating", "numpy", "arrow", "torch"])
def test_top_n_saves_input_ranks(n, score_source):
    scores = np.array([0.1, np.nan, 0.9, 0.4], dtype=np.float32)
    items = ItemList(["a", "b", "c", "d"], scores=scores, rating=scores, rank=[1, 4, 7, 9])
    if score_source == "numpy":
        score_argument = scores
    elif score_source == "arrow":
        score_argument = pa.array(scores)
    elif score_source == "torch":
        score_argument = torch.from_numpy(scores)
    else:
        score_argument = score_source

    result = items.top_n(n, scores=score_argument, save_ranks=True)
    selected = np.array([2, 3, 0])
    if n is not None and n >= 0:
        selected = selected[:n]
    np.testing.assert_array_equal(result.ids(), items.ids()[selected])
    np.testing.assert_array_equal(result.field("prev_rank"), items.ranks()[selected])
    np.testing.assert_array_equal(result.ranks(), np.arange(1, len(selected) + 1))
    np.testing.assert_array_equal(result.scores(), scores[selected])
    assert result.ordered
    assert items.field("prev_rank") is None
    np.testing.assert_array_equal(items.ranks(), [1, 4, 7, 9])


@pytest.mark.parametrize(
    "selector,selected",
    [
        (2, [2]),
        (slice(1, 3), [1, 2]),
        ([3, 0], [3, 0]),
        (np.array([True, False, True, False]), [0, 2]),
        (pa.array([2, 0], type=pa.int32()), [2, 0]),
        (np.array([], dtype=np.int32), []),
    ],
)
def test_take_saves_input_ranks(selector, selected):
    items = ItemList(["a", "b", "c", "d"], scores=[0.1, 0.2, 0.3, 0.4], rank=[1, 4, 7, 9])
    result = items.take(selector, save_ranks=True)
    np.testing.assert_array_equal(result.ids(), items.ids()[selected])
    np.testing.assert_array_equal(result.field("prev_rank"), items.ranks()[selected])
    np.testing.assert_array_equal(result.ranks(), np.arange(1, len(selected) + 1))
    assert items.field("prev_rank") is None

    default = items.take(selector)
    indexed = items[selector]
    np.testing.assert_array_equal(default.ids(), indexed.ids())
    np.testing.assert_array_equal(default.ranks(), indexed.ranks())
    assert default.field("prev_rank") is None


@pytest.mark.parametrize("with_vocabulary", [False, True])
@pytest.mark.parametrize("by_numbers", [False, True])
def test_remove_saves_ranks_including_fieldless_lists(with_vocabulary, by_numbers):
    vocabulary = Vocabulary(["a", "b", "c", "d"]) if with_vocabulary else None
    items = ItemList(
        ["a", "b", "c", "d"], item_nums=np.arange(4), vocabulary=vocabulary, ordered=True
    )
    removal = {"numbers": np.array([1, 3])} if by_numbers else {"ids": ["b", "d"]}
    result = items.remove(**removal, save_ranks=True)
    np.testing.assert_array_equal(result.ids(), ["a", "c"])
    np.testing.assert_array_equal(result.field("prev_rank"), [1, 3])
    np.testing.assert_array_equal(result.ranks(), [1, 2])
    assert result.vocabulary is vocabulary
    assert items.field("prev_rank") is None


def test_rank_preservation_uses_immediate_input_and_does_not_mutate_it():
    items = ItemList(["a", "b", "c"], scores=[0.1, 0.9, 0.4], ordered=True, prev_rank=[8, 9, 10])
    ranked = items.top_n(save_ranks=True)
    selected = ranked.take([2, 0], save_ranks=True)
    np.testing.assert_array_equal(ranked.field("prev_rank"), [2, 3, 1])
    np.testing.assert_array_equal(selected.field("prev_rank"), [3, 1])
    np.testing.assert_array_equal(items.field("prev_rank"), [8, 9, 10])
    np.testing.assert_array_equal(items.ids(), ["a", "b", "c"])


@pytest.mark.parametrize("operation", ["top_n", "take", "remove"])
def test_unordered_input_does_not_invent_previous_ranks(operation):
    items = ItemList(["a", "b", "c"], scores=[0.1, 0.9, 0.4])
    if operation == "top_n":
        result = items.top_n(2, save_ranks=True)
    elif operation == "take":
        result = items.take([2, 0], save_ranks=True)
    else:
        result = items.remove(ids=["b"], save_ranks=True)
    assert result.field("prev_rank") is None
    assert items.ranks() is None


def test_default_top_n_does_not_materialize_input_ranks():
    items = ItemList(["a", "b", "c"], scores=[0.1, 0.9, 0.4], ordered=True)
    assert items._ranks is None
    result = items.top_n(2)
    assert items._ranks is None
    assert result.field("prev_rank") is None
    np.testing.assert_array_equal(result.ids(), ["b", "c"])
