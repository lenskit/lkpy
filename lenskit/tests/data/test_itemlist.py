# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle

import numpy as np
import pandas as pd
import torch

from pytest import raises

from lenskit.data import ItemList
from lenskit.data.dataset import Dataset
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


def test_item_list_pickle_compact(ml_ds: Dataset):
    nums = [1, 0, 308, 24, 72]
    il = ItemList(item_nums=nums, vocabulary=ml_ds.items)
    assert len(il) == 5
    assert np.all(il.ids() == ml_ds.items.ids(nums))

    # check that pickling isn't very big (we don't pickle the vocabulary)
    data = pickle.dumps(il)
    print(len(data))
    assert len(data) <= 500

    il2 = pickle.loads(data)
    assert len(il2) == len(il)
    assert np.all(il2.ids() == il.ids())
    assert np.all(il2.numbers() == il.numbers())


def test_item_list_pickle_fields(ml_ds: Dataset):
    row = ml_ds.user_profile(user_num=400).item_list()
    assert row is not None
    data = pickle.dumps(row)
    r2 = pickle.loads(data)

    assert len(r2) == len(row)
    assert np.all(r2.ids() == row.ids())
    assert np.all(r2.numbers() == row.numbers())
    assert r2.field("rating") is not None
    assert np.all(r2.field("rating") == row.field("rating"))
    assert r2.field("timestamp") is not None
    assert np.all(r2.field("timestamp") == row.field("timestamp"))


def test_subset_mask(ml_ds: Dataset):
    row = ml_ds.user_profile(user_num=400).item_list()
    assert row is not None
    ratings = row.field("rating")
    assert ratings is not None

    mask = ratings > 3.0
    pos = row[mask]

    assert len(pos) == np.sum(mask)
    assert np.all(pos.ids() == row.ids()[mask])
    assert np.all(pos.numbers() == row.numbers()[mask])
    rf = row.field("rating")
    assert rf is not None
    prf = pos.field("rating")
    assert prf is not None
    assert np.all(prf == rf[mask])
    assert np.all(prf > 3.0)


def test_subset_idx(ml_ds: Dataset):
    row = ml_ds.user_profile(user_num=400).item_list()
    assert row is not None
    ratings = row.field("rating")
    assert ratings is not None

    ks = [0, 5, 15]
    pos = row[ks]

    assert len(pos) == 3
    assert np.all(pos.ids() == row.ids()[ks])
    assert np.all(pos.numbers() == row.numbers()[ks])
    rf = row.field("rating")
    assert rf is not None
    assert np.all(pos.field("rating") == rf[ks])


def test_subset_slice(ml_ds: Dataset):
    row = ml_ds.user_profile(user_num=400).item_list()
    assert row is not None
    ratings = row.field("rating")
    assert ratings is not None

    pos = row[5:10]

    assert len(pos) == 5
    assert np.all(pos.ids() == row.ids()[5:10])
    assert np.all(pos.numbers() == row.numbers()[5:10])
    rf = row.field("rating")
    assert rf is not None
    assert np.all(pos.field("rating") == rf[5:10])


def test_from_df():
    df = pd.DataFrame({"item_id": ITEMS, "item_num": np.arange(5), "score": np.random.randn(5)})
    il = ItemList.from_df(df, vocabulary=VOCAB)  # type: ignore
    assert len(il) == 5
    assert np.all(il.ids() == ITEMS)
    assert np.all(il.numbers() == np.arange(5))
    assert np.all(il.scores() == df["score"].values)


def test_from_df_user():
    df = pd.DataFrame(
        {"user_id": 50, "item_id": ITEMS, "item_num": np.arange(5), "score": np.random.randn(5)}
    )
    il = ItemList.from_df(df, vocabulary=VOCAB)  # type: ignore
    assert len(il) == 5
    assert np.all(il.ids() == ITEMS)
    assert np.all(il.numbers() == np.arange(5))
    assert np.all(il.scores() == df["score"].values)
    assert il.field("user_id") is None
