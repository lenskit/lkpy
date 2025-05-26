# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from pytest import mark, raises, warns

from lenskit.data import ItemList
from lenskit.data.collection import ItemListCollection, MutableItemListCollection, UserIDKey
from lenskit.data.collection._keys import create_key, project_key
from lenskit.data.dataset import Dataset
from lenskit.diagnostics import DataWarning
from lenskit.testing import DemoRecs, demo_recs

_log = logging.getLogger(__name__)


def test_generic_key():
    usk = create_key(("user_id", "seq_no"), "alphabet", 42)

    assert isinstance(usk, tuple)
    assert usk == ("alphabet", 42)
    assert usk.user_id == "alphabet"  # type: ignore
    assert usk.seq_no == 42  # type: ignore


def test_pickle_generic():
    usk = create_key(("user_id", "seq_no"), "alphabet", 42)

    bs = pickle.dumps(usk)
    usk2 = pickle.loads(bs)

    assert isinstance(usk2, tuple)
    assert usk2 == usk


def test_project_key():
    usk = create_key(("user_id", "seq_no"), "alphabet", 42)

    uk = project_key(usk, UserIDKey)
    assert isinstance(uk, UserIDKey)
    assert uk.user_id == "alphabet"
    assert uk == ("alphabet",)


def test_project_missing_fails():
    usk = create_key(("seq_no",), 42)

    with raises(TypeError, match="missing field"):
        project_key(usk, UserIDKey)


def test_collection_empty():
    empty = ItemListCollection.empty(UserIDKey)
    assert len(empty) == 0
    with raises(StopIteration):
        next(iter(empty))


def test_collection_base_ctor():
    empty = ItemListCollection(UserIDKey)  # type: ignore
    assert len(empty) == 0
    with raises(StopIteration):
        next(iter(empty))


def test_collection_mutable_ctor():
    empty = MutableItemListCollection(UserIDKey)  # type: ignore
    assert len(empty) == 0
    with raises(StopIteration):
        next(iter(empty))


def test_collection_add_item():
    ilc = ItemListCollection.empty(UserIDKey)

    ilc.add(ItemList(["a"]), 72)
    assert len(ilc) == 1
    key, il = next(iter(ilc))
    assert key.user_id == 72
    assert np.all(il.ids() == ["a"])

    il2 = ilc.lookup(72)
    assert il2 is il


def test_collection_from_dict_nt_key():
    ilc = ItemListCollection.from_dict({UserIDKey(72): ItemList(["a"])}, key=UserIDKey)

    assert len(ilc) == 1
    k, il = ilc[0]
    assert k.user_id == 72
    assert len(il) == 1
    assert np.all(il.ids() == ["a"])


def test_collection_from_dict_tuple_key():
    ilc = ItemListCollection.from_dict({(72,): ItemList(["a"])}, key=UserIDKey)

    assert len(ilc) == 1
    k, il = ilc[0]
    assert k.user_id == 72
    assert len(il) == 1
    assert np.all(il.ids() == ["a"])


def test_collection_from_dict_singleton_key():
    ilc = ItemListCollection.from_dict({72: ItemList(["a"])}, key=UserIDKey)

    assert len(ilc) == 1
    k, il = ilc[0]
    assert k.user_id == 72
    assert len(il) == 1
    assert np.all(il.ids() == ["a"])


def test_collection_from_dict_singleton_field():
    ilc = ItemListCollection.from_dict({72: ItemList(["a"])}, key="user_id")

    assert len(ilc) == 1
    k, il = ilc[0]
    assert k.user_id == 72  # type: ignore
    assert len(il) == 1
    assert np.all(il.ids() == ["a"])


def test_lookup_projected():
    ilc = ItemListCollection.from_dict({72: ItemList(["a"])}, key="user_id")

    usk = create_key(("user_id", "seq"), 72, 100)
    il = ilc.lookup_projected(usk)
    assert il is not None
    assert len(il) == 1
    assert np.all(il.ids() == ["a"])


def test_add_from():
    ilc = ItemListCollection.empty(["model", "user_id"])

    ilc1 = ItemListCollection.from_dict({72: ItemList(["a", "b"]), 48: ItemList()}, key="user_id")
    ilc.add_from(ilc1, model="foo")

    assert len(ilc) == 2
    il = ilc.lookup(("foo", 72))
    assert il is not None
    assert il.ids().tolist() == ["a", "b"]
    il = ilc.lookup(("foo", 48))
    assert il is not None
    assert len(il) == 0


def test_from_df(rng, ml_ratings: pd.DataFrame):
    ml_ratings = ml_ratings.rename(columns={"user": "user_id", "item": "item_id"})
    ilc = ItemListCollection.from_df(ml_ratings, UserIDKey)
    assert len(ilc) == ml_ratings["user_id"].nunique()
    assert set(k.user_id for k in ilc.keys()) == set(ml_ratings["user_id"])

    for uid in rng.choice(ml_ratings["user_id"].unique(), 25):
        items = ilc.lookup(user_id=uid)
        udf = ml_ratings[ml_ratings["user_id"] == uid]
        assert len(items) == len(udf)
        assert np.all(np.unique(items.ids()) == np.unique(udf["item_id"]))

    tot = sum(len(il) for il in ilc.lists())
    assert tot == len(ml_ratings)


def test_from_df_auto(rng, ml_ratings: pd.DataFrame):
    ml_ratings = ml_ratings.rename(columns={"user": "user_id", "item": "item_id"})
    with warns(DataWarning, match="inferring"):
        ilc = ItemListCollection.from_df(ml_ratings)

    assert len(ilc) == ml_ratings["user_id"].nunique()
    assert set(k.user_id for k in ilc.keys()) == set(ml_ratings["user_id"])

    for uid in rng.choice(ml_ratings["user_id"].unique(), 25):
        items = ilc.lookup(user_id=uid)
        udf = ml_ratings[ml_ratings["user_id"] == uid]
        assert len(items) == len(udf)
        assert np.all(np.unique(items.ids()) == np.unique(udf["item_id"]))

    tot = sum(len(il) for il in ilc.lists())
    assert tot == len(ml_ratings)


def test_to_df():
    ilc = ItemListCollection.from_dict(
        {72: ItemList(["a"], scores=[1]), 82: ItemList(["a", "b", "c"], scores=[3, 4, 10])},
        key="user_id",
    )
    df = ilc.to_df()
    print(df)
    assert len(df) == 4
    assert df["user_id"].tolist() == [72, 82, 82, 82]
    assert list(df.columns) == ["user_id", "item_id", "score"]


def test_to_df_no_nums():
    ilc = ItemListCollection.from_dict(
        {
            72: ItemList(["a"], item_nums=[0], scores=[1]),
            82: ItemList(["a", "b", "c"], item_nums=np.arange(3), scores=[3, 4, 10]),
        },
        key="user_id",
    )
    df = ilc.to_df()
    print(df)
    assert len(df) == 4
    assert df["user_id"].tolist() == [72, 82, 82, 82]
    assert list(df.columns) == ["user_id", "item_id", "score"]


def test_to_df_warn_empty():
    ilc = ItemListCollection.from_dict(
        {
            72: ItemList(["a"], scores=[1]),
            40: ItemList(),
            82: ItemList(["a", "b", "c"], scores=[3, 4, 10]),
        },
        key="user_id",
    )
    with warns(DataWarning, match="dropped"):
        df = ilc.to_df()
    print(df)
    assert len(df) == 4
    assert df["user_id"].tolist() == [72, 82, 82, 82]


def test_to_arrow():
    ilc = ItemListCollection.from_dict(
        {72: ItemList(["a"], scores=[1]), 82: ItemList(["a", "b", "c"], scores=[3, 4, 10])},
        key="user_id",
    )
    tbl = ilc.to_arrow()
    print(tbl)
    assert tbl.num_rows == 2
    assert np.all(tbl.column("user_id").to_numpy() == [72, 82])
    il_field = tbl.field("items")
    assert pa.types.is_list(il_field.type)
    assert isinstance(il_field.type, pa.ListType)
    il_type = il_field.type.value_type
    assert pa.types.is_struct(il_type)
    assert isinstance(il_type, pa.StructType)
    names = [il_type.field(i).name for i in range(il_type.num_fields)]
    assert names == ["item_id", "score"]


def test_to_arrow_flat():
    ilc = ItemListCollection.from_dict(
        {72: ItemList(["a"], scores=[1]), 82: ItemList(["a", "b", "c"], scores=[3, 4, 10])},
        key="user_id",
    )
    tbl = ilc.to_arrow(layout="flat")
    print(tbl)
    assert tbl.num_columns == 3
    assert tbl.num_rows == 4
    assert np.all(tbl.column("user_id").to_numpy() == [72, 82, 82, 82])
    assert np.all(tbl.column("item_id").to_numpy() == ["a", "a", "b", "c"])
    assert np.all(tbl.column("score").to_numpy() == [1, 3, 4, 10])


@mark.parametrize("layout", ["native", "flat"])
def test_save_parquet(ml_ds: Dataset, tmpdir: Path, layout):
    ilc = ItemListCollection.empty(["user_id"])
    for user in ml_ds.users.ids():
        ilc.add(ml_ds.user_row(user), user_id=user)

    _log.info("initial list:\n%s", ilc.to_df())

    f = tmpdir / "items.parquet"
    ilc.save_parquet(f, layout=layout)

    assert f.exists()

    ilc2 = ItemListCollection.load_parquet(f, layout=layout)
    _log.info("loaded list:\n%s", ilc2.to_df())
    assert len(ilc2) == len(ilc)
    assert set(ilc2.keys()) == set(ilc.keys())

    il1 = next(ilc.lists())
    _log.info("first item (initial):\n%s", il1.to_df())
    il2 = next(ilc2.lists())
    _log.info("first item (loaded):\n%s", il2.to_df())
    assert len(il1) == len(il2)

    assert sum(len(l1) for l1 in ilc2.lists()) == sum(len(l2) for l2 in ilc.lists())


@mark.filterwarnings("ignore:.*dropped.*:lenskit.diagnostics.DataWarning")
def test_save_parquet_with_empty(ml_ds: Dataset, tmpdir: Path):
    ilc = ItemListCollection.empty(["user_id"])
    ilc.add(ItemList(), user_id=-1)
    for user in ml_ds.users.ids():
        if user % 348:
            ilc.add(ml_ds.user_row(user), user_id=user)
        else:
            ilc.add(ItemList(), user_id=user)

    _log.info("initial list:\n%s", ilc.to_df())

    f = tmpdir / "items.parquet"
    ilc.save_parquet(f)

    assert f.exists()

    ilc2 = ItemListCollection.load_parquet(f)
    _log.info("loaded list:\n%s", ilc2.to_df())
    assert len(ilc2) == len(ilc)
    assert set(ilc2.keys()) == set(ilc.keys())

    il1 = next(ilc.lists())
    _log.info("first item (initial):\n%s", il1.to_df())
    il2 = next(ilc2.lists())
    _log.info("first item (loaded):\n%s", il2.to_df())
    assert len(il1) == len(il2)

    assert sum(len(l1) for l1 in ilc2.lists()) == sum(len(l2) for l2 in ilc.lists())


def test_save_parquet_with_mkdir(tmpdir: Path):
    ilc = ItemListCollection.empty(["user_id"])

    f = tmpdir / "subdir" / "items.parquet"
    ilc.save_parquet(f, mkdir=True)
    assert (tmpdir / "subdir").exists()

    f_no_mkdir = tmpdir / "no_mkdir" / "items.parquet"
    ilc.save_parquet(f_no_mkdir, mkdir=False)
    assert not (tmpdir / "no_mkdir").exists()


def test_write_recs_parquet(demo_recs, tmpdir: Path):
    split, recs = demo_recs

    test_f = tmpdir / "test.parquet"
    rec_f = tmpdir / "recs.parquet"

    split.test.save_parquet(test_f)
    recs.save_parquet(rec_f)

    t2 = ItemListCollection.load_parquet(test_f)
    assert list(t2.keys()) == list(split.test.keys())

    r2 = ItemListCollection.load_parquet(rec_f)
    assert list(r2.keys()) == list(recs.keys())
    assert all(il.ordered for il in r2.lists())


def test_recs_df_expected_column(demo_recs: DemoRecs):
    rec_df = demo_recs.recommendations.to_df()
    print(rec_df)
    print(demo_recs.recommendations[0])
    assert list(rec_df.columns) == ["user_id", "item_id", "score", "rank"]


def test_to_dataset(demo_recs: DemoRecs):
    test = demo_recs.split.test
    test_ds = test.to_dataset()
    assert test_ds.user_count == len(test)
    assert set(test_ds.users.ids()) == set(k.user_id for k in test.keys())

    test_df = test.to_df()
    assert test_ds.interaction_count == len(test_df)

    src_user_counts = test_df.groupby("user_id")["item_id"].count()
    src_item_counts = test_df.groupby("item_id")["user_id"].count()

    ds_user_counts = test_ds.user_stats()["count"]
    ds_item_counts = test_ds.item_stats()["count"]

    src_user_counts, ds_user_counts = src_user_counts.align(ds_user_counts, fill_value=0)
    src_item_counts, ds_item_counts = src_item_counts.align(ds_item_counts, fill_value=0)

    assert np.all(ds_user_counts == src_user_counts)
    assert np.all(ds_item_counts == src_item_counts)
