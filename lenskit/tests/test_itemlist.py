# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import torch

from pytest import raises

from lenskit.data import ItemList
from lenskit.data.vocab import Vocabulary

ITEMS = ["a", "b", "c", "d", "e"]
VOCAB = Vocabulary(ITEMS)


def test_empty():
    il = ItemList()

    assert len(il) == 0
    assert il.numbers().shape == (0,)
    assert il.ids().shape == (0,)
    assert il.scores() is None


def test_item_list():
    il = ItemList(item_ids=["one", "two"])

    assert len(il) == 2
    assert il.ids().shape == (2,)

    with raises(RuntimeError, match="item numbers not available"):
        il.numbers()


def test_item_list_alias():
    il = ItemList(item_id=["one", "two"])

    assert len(il) == 2
    assert il.ids().shape == (2,)

    with raises(RuntimeError, match="item numbers not available"):
        il.numbers()


def test_item_list_bad_type():
    with raises(TypeError):
        ItemList(item_id=[3.4, 7.2])


def test_item_list_bad_dimension():
    with raises(TypeError):
        ItemList(item_id=[["one", "two"]])


def test_item_num_array():
    il = ItemList(item_nums=np.arange(5))

    assert len(il) == 5
    assert il.numbers().shape == (5,)

    with raises(RuntimeError, match="item IDs not available"):
        il.ids()


def test_item_num_alias():
    il = ItemList(item_num=np.arange(5))

    assert len(il) == 5
    assert il.numbers().shape == (5,)

    with raises(RuntimeError, match="item IDs not available"):
        il.ids()


def test_item_num_bad_type():
    with raises(TypeError):
        ItemList(item_num=np.random.randn(5))


def test_item_num_bad_dims():
    with raises(TypeError):
        ItemList(item_num=[[1, 3, 8, 4]])


def test_item_ids_num_mismatch_sizes():
    with raises(TypeError, match="has incorrect shape"):
        ItemList(item_ids=ITEMS, item_num=np.arange(4))


def test_item_num_list_vocab():
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB)

    assert len(il) == 5
    assert il.numbers().shape == (5,)
    assert il.ids().shape == (5,)

    assert all(il.numbers() == np.arange(5))
    assert all(il.ids() == ITEMS)


def test_item_id_list_vocab():
    il = ItemList(item_ids=ITEMS, vocabulary=VOCAB)

    assert len(il) == 5
    assert il.numbers().shape == (5,)
    assert il.ids().shape == (5,)

    assert all(il.numbers() == np.arange(5))
    assert all(il.ids() == ITEMS)


def test_scores():
    data = np.random.randn(5)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)

    scores = il.scores()
    assert scores is not None
    assert scores.shape == (5,)
    assert np.all(scores == data)

    st = il.scores("torch")
    assert isinstance(st, torch.Tensor)
    assert st.shape == (5,)
    assert np.all(st.numpy() == data)

    assert il.ranks() is None


def test_scores_pandas_no_index():
    data = np.random.randn(5)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)

    scores = il.scores("pandas")
    assert scores is not None
    assert scores.shape == (5,)
    assert np.all(scores == data)


def test_scores_pandas_id_index():
    data = np.random.randn(5)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)
    scores = il.scores("pandas", index="ids")
    assert scores is not None
    assert scores.shape == (5,)
    assert np.all(scores == data)
    assert scores.index.name == "item_id"
    assert np.all(scores.index.values == ITEMS)


def test_scores_pandas_num_index():
    data = np.random.randn(5)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)
    scores = il.scores("pandas", index="numbers")
    assert scores is not None
    assert scores.shape == (5,)
    assert np.all(scores == data)
    assert scores.index.name == "item_num"
    assert np.all(scores.index.values == np.arange(5))


def test_ranks():
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, ordered=True)
    assert il.ordered

    ranks = il.ranks()
    assert ranks is not None
    assert ranks.shape == (5,)
    assert np.all(ranks == np.arange(1, 6))


def test_numbers_alt_vocab():
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB)

    av = Vocabulary(["A", "B"] + ITEMS)
    nums = il.numbers(vocabulary=av)
    assert np.all(nums == np.arange(2, 7))


def test_pandas_df():
    data = np.random.randn(5)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)

    df = il.to_df()
    assert np.all(df["item_id"] == ITEMS)
    assert np.all(df["item_num"] == np.arange(5))
    assert np.all(df["score"] == data)
    assert "rank" not in df.columns


def test_pandas_df_ordered():
    data = np.random.randn(5)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data, ordered=True)

    df = il.to_df()
    assert np.all(df["item_id"] == ITEMS)
    assert np.all(df["item_num"] == np.arange(5))
    assert np.all(df["score"] == data)
    assert np.all(df["rank"] == np.arange(1, 6))
