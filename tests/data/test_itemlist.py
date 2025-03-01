# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import torch

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given, settings
from pytest import mark, raises, warns

from lenskit.data import Dataset, ItemList
from lenskit.data.vocab import Vocabulary
from lenskit.diagnostics import DataWarning

ITEMS = ["a", "b", "c", "d", "e"]
VOCAB = Vocabulary(ITEMS)


def test_empty():
    il = ItemList()

    assert len(il) == 0
    assert il.numbers().shape == (0,)
    assert il.ids().shape == (0,)
    assert il.scores() is None


def test_empty_ids():
    il = ItemList(item_ids=[])
    assert len(il) == 0
    assert il.ids().shape == (0,)
    assert il.scores() is None


def test_empty_nums():
    il = ItemList(item_nums=[])
    assert len(il) == 0
    assert il.numbers().shape == (0,)
    assert il.scores() is None


def test_empty_list():
    il = ItemList([])
    assert len(il) == 0
    assert il.ids().shape == (0,)
    assert il.scores() is None


def test_item_list():
    il = ItemList(item_ids=["one", "two"])

    assert len(il) == 2
    assert il.ids().shape == (2,)

    with raises(RuntimeError, match="item numbers not available"):
        il.numbers()


def test_item_list_ctor():
    il = ItemList(["one", "two"])

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
    data = np.random.randn(5).astype(np.float32)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)

    scores = il.scores()
    assert scores is not None
    assert scores.shape == (5,)
    assert np.all(scores == data.astype(np.float32))
    assert scores.dtype == np.float32

    st = il.scores("torch")
    assert isinstance(st, torch.Tensor)
    assert st.shape == (5,)
    assert np.all(st.numpy() == data)

    assert il.ranks() is None


def test_rank_field():
    data = np.random.randn(5).astype(np.float32)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data, rank=np.arange(5) + 1)
    assert il.ordered
    assert np.all(il.ranks() == np.arange(5) + 1)


def test_torch_rank_field():
    data = np.random.randn(5).astype(np.float32)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data, rank=torch.arange(5) + 1)
    assert il.ordered
    assert np.all(il.ranks() == np.arange(5) + 1)


def test_ordered_mismatch():
    data = np.random.randn(5).astype(np.float32)
    with warns(DataWarning, match="ordered=False"):
        il = ItemList(
            item_nums=np.arange(5),
            ordered=False,
            vocabulary=VOCAB,
            scores=data,
            rank=np.arange(5) + 1,
        )
    assert not il.ordered
    assert il.ranks() is None


def test_rank_not_one():
    data = np.random.randn(5).astype(np.float32)
    with warns(DataWarning, match="begin with 1"):
        il = ItemList(
            item_nums=np.arange(5), vocabulary=VOCAB, scores=data, rank=np.arange(2, 11, 2)
        )
    assert il.ordered
    assert np.all(il.ranks() == np.arange(2, 11, 2))


def test_scores_pandas_no_index():
    data = np.random.randn(5).astype(np.float32)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)

    scores = il.scores("pandas")
    assert scores is not None
    assert scores.shape == (5,)
    assert np.all(scores == data)


def test_scores_pandas_id_index():
    data = np.random.randn(5).astype(np.float32)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)
    scores = il.scores("pandas", index="ids")
    assert scores is not None
    assert scores.shape == (5,)
    assert np.all(scores == data)
    assert scores.index.name == "item_id"
    assert np.all(scores.index.values == ITEMS)


def test_scores_pandas_num_index():
    data = np.random.randn(5).astype(np.float32)
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


@mark.parametrize("src", ["ids", "numbers"])
def test_numbers_numpy(src):
    match src:
        case "ids":
            il = ItemList(item_ids=ITEMS, vocabulary=VOCAB)
        case "numbers":
            il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB)

    nums = il.numbers(format="numpy")
    assert isinstance(nums, np.ndarray)
    assert np.all(nums == np.arange(5))

    nums = il.numbers(format="numpy", vocabulary=Vocabulary(ITEMS))
    assert isinstance(nums, np.ndarray)
    assert np.all(nums == np.arange(5))


@mark.parametrize("src", ["ids", "numbers"])
def test_numbers_torch(src):
    match src:
        case "ids":
            il = ItemList(item_ids=ITEMS, vocabulary=VOCAB)
        case "numbers":
            il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB)

    nums = il.numbers(format="torch")
    assert torch.is_tensor(nums)
    assert torch.all(nums == torch.arange(5))

    nums = il.numbers(format="torch", vocabulary=Vocabulary(ITEMS))
    assert torch.is_tensor(nums)
    assert torch.all(nums == torch.arange(5))


@mark.parametrize("src", ["ids", "numbers"])
def test_numbers_arrow(src):
    match src:
        case "ids":
            il = ItemList(item_ids=ITEMS, vocabulary=VOCAB)
        case "numbers":
            il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB)

    nums = il.numbers(format="arrow")
    assert isinstance(nums, pa.Int32Array)
    assert np.all(nums.to_numpy() == np.arange(5))

    nums = il.numbers(format="arrow", vocabulary=Vocabulary(ITEMS))
    assert isinstance(nums, pa.Int32Array)
    assert np.all(nums.to_numpy() == np.arange(5))


def test_pandas_df():
    data = np.random.randn(5).astype(np.float32)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)

    df = il.to_df()
    assert np.all(df["item_id"] == ITEMS)
    assert np.all(df["item_num"] == np.arange(5))
    assert np.all(df["score"] == data)
    assert "rank" not in df.columns


def test_pandas_df_ordered():
    data = np.random.randn(5).astype(np.float32)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data, ordered=True)

    df = il.to_df()
    assert np.all(df["item_id"] == ITEMS)
    assert np.all(df["item_num"] == np.arange(5))
    assert np.all(df["score"] == data)
    assert np.all(df["rank"] == np.arange(1, 6))


def test_pandas_df_no_numbers():
    data = np.random.randn(5).astype(np.float32)
    il = ItemList(item_ids=ITEMS, vocabulary=VOCAB, scores=data, ordered=True)
    df = il.to_df(numbers=False)
    assert "item_id" in df.columns

    # even with a vocabulary, we should have no numbers
    assert "item_num" not in df.columns


def test_arrow_table():
    data = np.random.randn(5).astype(np.float32)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)

    tbl = il.to_arrow(numbers=True)
    assert isinstance(tbl, pa.Table)
    assert tbl.num_columns == 3
    assert np.all(tbl.column("item_id").to_numpy() == ITEMS)
    assert np.all(tbl.column("item_num").to_numpy() == np.arange(5))
    assert np.all(tbl.column("score").to_numpy() == data)


def test_arrow_array():
    data = np.random.randn(5).astype(np.float32)
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data)

    tbl = il.to_arrow(numbers=True, type="array")
    assert isinstance(tbl, pa.StructArray)
    tbl = pa.Table.from_struct_array(tbl)
    assert tbl.num_columns == 3
    assert np.all(tbl.column("item_id").to_numpy() == ITEMS)
    assert np.all(tbl.column("item_num").to_numpy() == np.arange(5))
    assert np.all(tbl.column("score").to_numpy() == data)


def test_empty_arrow():
    il = ItemList()

    arr = il.to_arrow()
    assert arr.num_rows == 0


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
    row = ml_ds.user_row(user_num=400)
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
    row = ml_ds.user_row(user_num=400)
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


def test_subset_idx_list(ml_ds: Dataset):
    row = ml_ds.user_row(user_num=400)
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


def test_subset_idx_array(ml_ds: Dataset):
    row = ml_ds.user_row(user_num=400)
    assert row is not None
    ratings = row.field("rating")
    assert ratings is not None

    ks = np.array([0, 5, 15])
    pos = row[ks]

    assert len(pos) == 3
    assert np.all(pos.ids() == row.ids()[ks])
    assert np.all(pos.numbers() == row.numbers()[ks])
    rf = row.field("rating")
    assert rf is not None
    assert np.all(pos.field("rating") == rf[ks])


@given(st.data())
def test_subset_index_array_stress(ml_ds_unchecked: Dataset, data: st.DataObject):
    ml_ds = ml_ds_unchecked
    uno = data.draw(st.integers(0, ml_ds.user_count - 1))
    row = ml_ds.user_row(user_num=uno)
    assert row is not None
    ratings = row.field("rating")
    assert ratings is not None

    ks = data.draw(
        nph.arrays(np.int32, st.integers(0, 1000), elements=st.integers(0, len(row) - 1))
    )
    found = row[ks]

    assert len(found) == len(ks)
    assert np.all(found.ids() == row.ids()[ks])
    assert np.all(found.numbers() == row.numbers()[ks])
    rf = row.field("rating")
    assert rf is not None
    assert np.all(found.field("rating") == rf[ks])


def test_subset_slice(ml_ds: Dataset):
    row = ml_ds.user_row(user_num=400)
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
    df = pd.DataFrame(
        {"item_id": ITEMS, "item_num": np.arange(5), "score": np.random.randn(5).astype(np.float32)}
    )
    il = ItemList.from_df(df, vocabulary=VOCAB)  # type: ignore
    assert len(il) == 5
    assert np.all(il.ids() == ITEMS)
    assert np.all(il.numbers() == np.arange(5))
    assert np.all(il.scores() == df["score"].values)
    assert not il.ordered


def test_from_df_user():
    df = pd.DataFrame(
        {
            "user_id": 50,
            "item_id": ITEMS,
            "item_num": np.arange(5),
            "score": np.random.randn(5).astype(np.float32),
        }
    )
    il = ItemList.from_df(df, vocabulary=VOCAB)  # type: ignore
    assert len(il) == 5
    assert np.all(il.ids() == ITEMS)
    assert np.all(il.numbers() == np.arange(5))
    assert np.all(il.scores() == df["score"].values)
    assert il.field("user_id") is None


def test_from_df_ranked():
    df = pd.DataFrame(
        {
            "item_id": ITEMS,
            "item_num": np.arange(5),
            "rank": np.arange(1, 6),
            "score": np.random.randn(5).astype(np.float32),
        }
    )
    il = ItemList.from_df(df, vocabulary=VOCAB)  # type: ignore
    assert len(il) == 5
    assert np.all(il.ids() == ITEMS)
    assert np.all(il.numbers() == np.arange(5))
    assert np.all(il.scores() == df["score"].values)
    assert il.ordered
    assert np.all(il.ranks() == np.arange(1, 6))


def test_from_arrow_table():
    df = pd.DataFrame(
        {"item_id": ITEMS, "item_num": np.arange(5), "score": np.random.randn(5).astype(np.float32)}
    )
    arr = pa.Table.from_pandas(df)
    il = ItemList.from_arrow(arr, vocabulary=VOCAB)  # type: ignore
    assert len(il) == 5
    assert np.all(il.ids() == ITEMS)
    assert np.all(il.numbers() == np.arange(5))
    assert np.all(il.scores() == df["score"].values)
    assert not il.ordered


def test_from_arrow_table_nullable():
    df = pd.DataFrame(
        {
            "item_id": np.arange(1, 100),
            "item_num": np.arange(99),
            "score": np.random.randn(99).astype(np.float32),
        }
    )
    df.loc[df["score"] < 0, "score"] = np.nan
    arr = pa.Table.from_pandas(df)
    il = ItemList.from_arrow(arr)  # type: ignore
    assert len(il) == 99
    assert np.all(il.ids() == np.arange(1, 100))
    scores = il.scores()
    assert scores is not None
    assert np.all(np.isfinite(scores) == df["score"].notnull())
    assert np.all(scores[df["score"].notnull()] == df.loc[df["score"].notnull(), "score"])
    assert not il.ordered


def test_from_empty_arrow():
    tbl = pa.table(
        {"item_num": pa.array([], type=pa.int32()), "score": pa.array([], type=pa.float32())}
    )
    print(tbl)
    il = ItemList.from_arrow(tbl, vocabulary=VOCAB)
    assert len(il) == 0
    scores = il.scores()
    assert scores is not None
    assert len(scores) == 0


def test_from_arrow_array():
    df = pd.DataFrame(
        {"item_id": ITEMS, "item_num": np.arange(5), "score": np.random.randn(5).astype(np.float32)}
    )
    arr = pa.Table.from_pandas(df).to_struct_array()
    il = ItemList.from_arrow(arr, vocabulary=VOCAB)  # type: ignore
    assert len(il) == 5
    assert np.all(il.ids() == ITEMS)
    assert np.all(il.numbers() == np.arange(5))
    assert np.all(il.scores() == df["score"].values)
    assert not il.ordered


def test_copy_ctor():
    data = np.random.randn(5).astype(np.float32)
    extra = np.random.randn(5).astype(np.float32)

    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data, extra=extra)
    copy = ItemList(il)
    assert copy is not il
    assert len(copy) == len(il)

    assert np.all(copy.ids() == il.ids())
    assert np.all(copy.numbers() == il.numbers())
    assert np.all(copy.scores() == data)
    assert copy._vocab is il._vocab

    x = copy.field("extra")
    assert x is not None
    assert np.all(x == extra)


def test_copy_na_scores():
    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB)
    il2 = ItemList(il, scores=np.nan)

    scores = il2.scores()
    assert scores is not None
    assert np.all(np.isnan(scores))


def test_copy_ctor_remove_scores():
    data = np.random.randn(5).astype(np.float32)
    extra = np.random.randn(5).astype(np.float32)

    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data, extra=extra)
    copy = ItemList(il, scores=False)
    assert copy is not il
    assert len(copy) == len(il)

    assert np.all(copy.ids() == il.ids())
    assert np.all(copy.numbers() == il.numbers())
    assert copy.scores() is None
    assert copy._vocab is il._vocab

    x = copy.field("extra")
    assert x is not None
    assert np.all(x == extra)


def test_copy_ctor_remove_extra():
    data = np.random.randn(5).astype(np.float32)
    extra = np.random.randn(5).astype(np.float32)

    il = ItemList(item_nums=np.arange(5), vocabulary=VOCAB, scores=data, extra=extra)
    copy = ItemList(il, extra=False)
    assert copy is not il
    assert len(copy) == len(il)

    assert np.all(copy.ids() == il.ids())
    assert np.all(copy.numbers() == il.numbers())
    assert np.all(copy.scores() == data)
    assert copy._vocab is il._vocab

    x = copy.field("extra")
    assert x is None


def test_from_vocab(ml_ds: Dataset):
    items = ItemList.from_vocabulary(ml_ds.items)

    assert len(items) == ml_ds.item_count
    assert np.all(items.ids() == ml_ds.items.ids())
    assert np.all(items.numbers() == np.arange(len(items)))
    assert items.vocabulary is ml_ds.items
