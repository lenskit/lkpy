# pyright: strict
import numpy as np
import pandas as pd
import pyarrow as pa

from pytest import raises

from lenskit.data import DatasetBuilder
from lenskit.diagnostics import DataError


def test_empty_builder():
    dsb = DatasetBuilder()
    assert not dsb.relationship_classes()
    ecs = dsb.entity_classes()
    assert len(ecs) == 1
    items = ecs["item"]
    assert items is not None
    assert items.id_type is None
    assert not items.attributes
    assert dsb.record_count("item") == 0
    with raises(KeyError):
        dsb.record_count("user")


def test_add_entity_ids():
    dsb = DatasetBuilder()

    dsb.add_entities("item", ["a", "b", "x", "y", "z"])
    icls = dsb.entity_classes()["item"]
    assert icls.id_type == "str"
    assert dsb.entity_id_type("item") == pa.string()

    ds = dsb.build()
    assert ds.item_count == 5
    assert np.all(ds.items.ids() == ["a", "b", "x", "y", "z"])
    assert np.all(ds.entities("item").ids() == ["a", "b", "x", "y", "z"])


def test_add_new_entity_ids():
    dsb = DatasetBuilder()

    dsb.add_entities("user", ["a", "b", "x", "y", "z"])
    ecs = dsb.entity_classes()
    assert set(ecs.keys()) == {"user", "item"}

    ucls = ecs["user"]
    assert ucls.id_type == "str"
    assert dsb.entity_id_type("user") == pa.string()

    ds = dsb.build()
    assert ds.item_count == 0
    assert ds.user_count == 5
    assert np.all(ds.users.ids() == ["a", "b", "x", "y", "z"])
    assert np.all(ds.entities("user").ids() == ["a", "b", "x", "y", "z"])


def test_add_duplicate_entities_forbidden():
    dsb = DatasetBuilder()

    dsb.add_entities("item", ["a", "b", "c"])
    with raises(DataError, match="duplicate"):
        dsb.add_entities("item", ["d", "b", "e"])


def test_add_duplicate_entities_overwrite():
    dsb = DatasetBuilder()

    dsb.add_entities("item", ["a", "b", "c"])
    dsb.add_entities("item", ["d", "b", "e"], duplicates="overwrite")

    ds = dsb.build()
    assert ds.item_count == 5
    assert set(ds.items.ids()) == {"a", "b", "c", "d", "e"}


def test_add_entities_upcast_existing():
    dsb = DatasetBuilder()

    dsb.add_entities("item", np.arange(10, dtype="i4"))
    assert dsb.entity_id_type("item") == pa.int32()

    dsb.add_entities("item", np.arange(100, 110, dtype="i8"))
    assert dsb.entity_id_type("item") == pa.int64()

    ds = dsb.build()
    assert ds.users.ids().dtype == np.int64


def test_add_entities_upcast_new():
    dsb = DatasetBuilder()

    dsb.add_entities("item", np.arange(10, dtype="i8"))
    assert dsb.entity_id_type("item") == pa.int64()

    dsb.add_entities("item", np.arange(100, 110, dtype="i4"))
    assert dsb.entity_id_type("item") == pa.int64()

    ds = dsb.build()
    assert ds.users.ids().dtype == np.int64


def test_reject_invalid_entity_id_type():
    dsb = DatasetBuilder()

    with raises(TypeError):
        dsb.add_entities("item", np.random.randn(10))  # type: ignore


def test_reject_duplicate_ids():
    dsb = DatasetBuilder()

    with raises(DataError):
        dsb.add_entities("item", ["a", "b", "a"])
