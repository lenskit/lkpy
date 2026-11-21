# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow as pa

from pytest import fixture, mark, raises

from lenskit.data import DatasetBuilder
from lenskit.diagnostics import DataError


class StreamTable:
    """Exercise the actual C stream protocol without a Polars dependency."""

    def __init__(self, table):
        self.table = table
        self.exports = 0

    def __arrow_c_stream__(self, requested_schema=None):
        self.exports += 1
        assert self.exports == 1, "a source must not be consumed twice"
        return self.table.__arrow_c_stream__(requested_schema)


@fixture(params=["arrow", "stream", "reader"])
def table_source(request):
    if request.param == "stream":
        return StreamTable
    if request.param == "reader":
        return lambda table: pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    return lambda table: table


def test_entity_table_with_scalar_and_list_attributes(table_source):
    table = pa.table({"item_id": ["b", "a"], "title": ["Beta", None], "genres": [["x", "y"], []]})
    builder = DatasetBuilder()
    builder.add_entities("item", table_source(table))
    dataset = builder.build()
    assert dataset.entities("item").arrow().to_pylist() == [
        {"item_id": "a", "title": None, "genres": []},
        {"item_id": "b", "title": "Beta", "genres": ["x", "y"]},
    ]
    # Conversion and sorting must not modify the caller's input.
    assert table.column("item_id").to_pylist() == ["b", "a"]


def test_interactions_with_timestamps_and_missing_values(table_source):
    table = pa.table(
        {
            "user_id": ["u2", "u1", "u1"],
            "item_id": ["b", "a", "b"],
            "rating": [4.0, None, 2.0],
            "timestamp": [1700000000, 1700000001, 1700000002],
        }
    )
    builder = DatasetBuilder()
    builder.add_interactions(
        "rating",
        table_source(table),
        entities=["user", "item"],
        missing="insert",
        timestamp_unit="s",
        default=True,
    )
    dataset = builder.build()
    result = dataset.interaction_table(format="arrow", original_ids=True).to_pylist()
    expected = [
        {**row, "timestamp": datetime.fromtimestamp(row["timestamp"], UTC).replace(tzinfo=None)}
        for row in table.to_pylist()
    ]
    assert sorted(result, key=lambda r: r["timestamp"]) == expected
    assert dataset.user_count == 2
    assert dataset.item_count == 2


def test_unknown_entities_are_filtered(table_source):
    builder = DatasetBuilder()
    builder.add_entities("user", [1])
    builder.add_entities("item", [2])
    builder.add_interactions(
        "click",
        table_source(pa.table({"user_id": [1, 3], "item_id": [2, 2]})),
        entities=["user", "item"],
        missing="filter",
    )
    result = builder.build().interaction_table(format="arrow", original_ids=True)
    assert result.to_pylist() == [{"user_id": 1, "item_id": 2}]


def test_unknown_entities_still_raise(table_source):
    builder = DatasetBuilder()
    builder.add_entities("user", [1])
    builder.add_entities("item", [2])
    with raises(DataError, match="unknown IDs"):
        builder.add_interactions(
            "click",
            table_source(pa.table({"user_id": [3], "item_id": [2]})),
            entities=["user", "item"],
            missing="error",
        )


def test_scalar_attribute_table(table_source):
    builder = DatasetBuilder()
    builder.add_entities("item", [1, 2, 3])
    builder.add_scalar_attribute(
        "item", "title", table_source(pa.table({"item_id": [3, 1], "title": ["C", "A"]}))
    )
    assert builder.build().entities("item").arrow().column("title").to_pylist() == ["A", None, "C"]


def test_list_attribute_table(table_source):
    builder = DatasetBuilder()
    builder.add_entities("item", [1, 2, 3])
    builder.add_list_attribute(
        "item", "genres", table_source(pa.table({"item_id": [3, 1], "genres": [["z"], ["x", "y"]]}))
    )
    assert builder.build().entities("item").arrow().column("genres").to_pylist() == [
        ["x", "y"],
        None,
        ["z"],
    ]


def test_filter_with_table_source(table_source):
    builder = DatasetBuilder()
    builder.add_interactions(
        "click",
        {"user_id": np.array([1, 1, 2]), "item_id": np.array([1, 2, 2])},
        entities=["user", "item"],
        missing="insert",
    )
    builder.filter_interactions("click", remove=table_source(pa.table({"item_id": [2]})))
    assert builder.build().interaction_table(format="arrow", original_ids=True).to_pylist() == [
        {"user_id": 1, "item_id": 1}
    ]


def test_string_id_filter_with_existing_string_vocabulary(table_source):
    builder = DatasetBuilder()
    builder.add_entities("user", pa.array(["u1", "u2"], type=pa.string()))
    builder.add_entities("item", pa.array(["a", "b"], type=pa.string()))
    builder.add_interactions(
        "click",
        {"user_id": np.array(["u1", "u1", "u2"]), "item_id": np.array(["a", "b", "b"])},
        entities=["user", "item"],
    )
    builder.filter_interactions("click", remove=table_source(pa.table({"item_id": ["b"]})))
    assert builder.build().interaction_table(format="arrow", original_ids=True).to_pylist() == [
        {"user_id": "u1", "item_id": "a"}
    ]


def test_empty_stream_has_typed_entity_columns(table_source):
    table = pa.table(
        {"item_id": pa.array([], type=pa.string()), "title": pa.array([], type=pa.string())}
    )
    builder = DatasetBuilder()
    builder.add_entities("item", table_source(table))
    assert builder.build().item_count == 0


def test_pandas_index_ids_keep_existing_behavior():
    frame = pd.DataFrame({"title": ["B", "A"]}, index=pd.Index([2, 1], name="item_id"))
    builder = DatasetBuilder()
    builder.add_entities("item", frame)
    assert builder.build().entities("item").arrow().column("item_id").to_pylist() == [1, 2]
    assert frame.index.tolist() == [2, 1]


def test_pandas_arrow_series_ids_keep_existing_behavior():
    series = pd.Series([2, 1], dtype=pd.ArrowDtype(pa.int32()), name="item_id")
    builder = DatasetBuilder()
    builder.add_entities("item", series)
    assert builder.build().items.ids().tolist() == [1, 2]


@mark.parametrize("attribute", ["direct", "entity"])
@mark.parametrize("kind", ["float", "string"])
def test_fixed_size_list_attribute(table_source, attribute, kind):
    if kind == "float":
        values = [[0.1, 0.2], None, [0.3, 0.4]]
        value_type = pa.float64()
    else:
        values = [["a", "b"], None, ["c", "d"]]
        value_type = pa.string()
    frame = pa.table(
        {"item_id": [3, 1, 2], "features": pa.array(values, type=pa.list_(value_type, 2))}
    )
    builder = DatasetBuilder()
    if attribute == "entity":
        builder.add_entities("item", table_source(frame))
    else:
        builder.add_entities("item", [1, 2, 3])
        builder.add_list_attribute("item", "features", table_source(frame))
    assert builder.build().entities("item").arrow().column("features").to_pylist() == [
        None,
        values[2],
        values[0],
    ]


@mark.parametrize("attribute", ["direct", "entity"])
def test_nullable_list_attribute(table_source, attribute):
    frame = pa.table({"item_id": [3, 1, 2], "genres": [["a", "b"], None, []]})
    builder = DatasetBuilder()
    if attribute == "entity":
        builder.add_entities("item", table_source(frame))
    else:
        builder.add_entities("item", [1, 2, 3])
        builder.add_list_attribute("item", "genres", table_source(frame))
    assert builder.build().entities("item").arrow().column("genres").to_pylist() == [
        None,
        [],
        ["a", "b"],
    ]


def test_empty_scalar_attribute_keeps_type(table_source):
    frame = pa.table(
        {"item_id": pa.array([], type=pa.string()), "title": pa.array([], type=pa.string())}
    )
    builder = DatasetBuilder()
    builder.add_entities("item", table_source(frame))
    result = builder.build().entities("item").arrow()
    assert result.field("title").type == pa.string()
