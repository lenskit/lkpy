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
