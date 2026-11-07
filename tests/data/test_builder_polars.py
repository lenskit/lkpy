# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from datetime import UTC, datetime

from pytest import importorskip, mark, raises

from lenskit.data import Dataset, DatasetBuilder
from lenskit.diagnostics import DataError

pl = importorskip("polars")


def test_native_entity_attributes():
    frame = pl.DataFrame(
        {"item_id": [3, 1, 2], "title": ["C", None, "B"], "genres": [["x", "y"], [], None]}
    )
    original = frame.to_dicts()
    builder = DatasetBuilder()
    builder.add_entities("item", frame)
    assert builder.build().entities("item").arrow().to_pylist() == [
        {"item_id": 1, "title": None, "genres": []},
        {"item_id": 2, "title": "B", "genres": None},
        {"item_id": 3, "title": "C", "genres": ["x", "y"]},
    ]
    assert frame.to_dicts() == original


@mark.parametrize("attribute", ["direct", "entity"])
@mark.parametrize("kind", ["float", "string"])
def test_fixed_array_attribute(attribute, kind):
    if kind == "float":
        values = [[0.1, 0.2], None, [0.3, 0.4]]
        dtype = pl.Array(pl.Float64, 2)
    else:
        values = [["a", "b"], None, ["c", "d"]]
        dtype = pl.Array(pl.String, 2)
    frame = pl.DataFrame({"item_id": [3, 1, 2], "features": pl.Series(values, dtype=dtype)})
    builder = DatasetBuilder()
    if attribute == "entity":
        builder.add_entities("item", frame)
    else:
        builder.add_entities("item", [1, 2, 3])
        builder.add_list_attribute("item", "features", frame)
    assert builder.build().entities("item").arrow().column("features").to_pylist() == [
        None, values[2], values[0]
    ]
    assert frame["features"].to_list() == values


@mark.parametrize("attribute", ["direct", "entity"])
def test_binary_attribute(attribute):
    frame = pl.DataFrame(
        {"item_id": [3, 1, 2], "payload": pl.Series([b"\x00\xff", None, b""], dtype=pl.Binary)}
    )
    builder = DatasetBuilder()
    if attribute == "entity":
        builder.add_entities("item", frame)
    else:
        builder.add_entities("item", [1, 2, 3])
        builder.add_scalar_attribute("item", "payload", frame)
    assert builder.build().entities("item").arrow().column("payload").to_pylist() == [
        None, b"", b"\x00\xff"
    ]


def test_native_interactions_timestamps_and_filtering():
    frame = pl.DataFrame(
        {"user_id": ["u2", "u1", "u1"], "item_id": ["b", "a", "b"],
         "rating": [4.0, None, 2.0], "timestamp": [1700000000, 1700000001, 1700000002]}
    )
    builder = DatasetBuilder()
    builder.add_interactions(
        "rating", frame, entities=["user", "item"], missing="insert", timestamp_unit="s"
    )
    builder.filter_interactions("rating", remove=pl.DataFrame({"item_id": ["b"]}))
    assert builder.build().interaction_table(format="arrow", original_ids=True).to_pylist() == [
        {"user_id": "u1", "item_id": "a", "rating": None,
         "timestamp": datetime.fromtimestamp(1700000001, UTC).replace(tzinfo=None)}
    ]


@mark.parametrize("missing", ["filter", "error"])
def test_unknown_entity_policy(missing):
    builder = DatasetBuilder()
    builder.add_entities("user", [1])
    builder.add_entities("item", [2])
    frame = pl.DataFrame({"user_id": [1, 3], "item_id": [2, 2]})
    if missing == "error":
        with raises(DataError, match="unknown IDs"):
            builder.add_interactions("click", frame, entities=["user", "item"], missing=missing)
    else:
        builder.add_interactions("click", frame, entities=["user", "item"], missing=missing)
        assert builder.build().interaction_table(format="arrow", original_ids=True).to_pylist() == [
            {"user_id": 1, "item_id": 2}
        ]


@mark.parametrize("ids", [[2, 1], ["b", "a"]])
def test_native_series_ids(ids):
    series = pl.Series("item_id", ids)
    builder = DatasetBuilder()
    builder.add_entities("item", series)
    assert builder.build().items.ids().tolist() == sorted(ids)
    assert series.to_list() == ids


def test_empty_typed_native_frame():
    frame = pl.DataFrame(schema={"item_id": pl.String, "title": pl.String,
                                 "features": pl.Array(pl.Float64, 2)})
    builder = DatasetBuilder()
    builder.add_entities("item", frame)
    dataset = builder.build()
    assert dataset.item_count == 0
    assert str(dataset.entities("item").arrow().field("title").type) == "large_string"


def test_native_frame_save_and_reload(tmp_path):
    frame = pl.DataFrame(
        {"item_id": [3, 1, 2], "features": pl.Series(
            [[0.1, 0.2], None, [0.3, 0.4]], dtype=pl.Array(pl.Float64, 2))}
    )
    builder = DatasetBuilder()
    builder.add_entities("item", frame)
    original = builder.build()
    original.save(tmp_path / "dataset")
    restored = Dataset.load(tmp_path / "dataset")
    assert restored.entities("item").arrow().to_pylist() == original.entities("item").arrow().to_pylist()
