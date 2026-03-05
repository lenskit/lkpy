# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: basic
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa

from pytest import approx, mark, raises, warns

from lenskit.data import Dataset, DatasetBuilder
from lenskit.data.schema import AllowableTroolean
from lenskit.diagnostics import DataError, DataWarning


def test_add_interactions_insert_ids_df():
    dsb = DatasetBuilder()

    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {"user_id": ["a", "a", "b", "c", "c", "c"], "item_id": ["x", "y", "z", "x", "y", "z"]}
        ),
        entities=["user", "item"],
        missing="insert",
    )

    ecs = dsb.entity_classes()
    assert set(ecs.keys()) == {"user", "item"}

    rcs = dsb.relationship_classes()
    assert set(rcs.keys()) == {"click"}
    print(dsb.schema.model_dump_json(indent=2))

    ucls = ecs["user"]
    assert ucls.id_type == "str"

    ds = dsb.build()
    assert ds.user_count == 3
    assert ds.item_count == 3
    assert np.all(ds.users.ids() == ["a", "b", "c"])
    assert np.all(ds.items.ids() == ["x", "y", "z"])

    istats = ds.item_stats()
    print(istats)
    assert np.all(istats["user_count"] == 2)

    log = ds.interaction_table(format="pandas", original_ids=True)
    assert isinstance(log, pd.DataFrame)
    print(log)
    assert all(log.columns == ["user_id", "item_id"])
    assert len(log) == 6

    mat = ds.interaction_matrix(format="structure")
    assert mat.nnz == 6
    assert np.all(mat.rowptrs == [0, 2, 3, 6])


def test_add_interactions_forbidden_attribute_name():
    dsb = DatasetBuilder()
    with raises(ValueError, match="invalid"):
        dsb.add_interactions(
            "click",
            pd.DataFrame(
                {
                    "user_id": ["a", "a", "b", "c", "c", "c"],
                    "item_id": ["x", "y", "z", "x", "y", "z"],
                    "_count": np.arange(1, 7),
                }
            ),
            entities=["user", "item"],
            missing="insert",
        )

    with raises(ValueError, match="invalid"):
        dsb.add_interactions(
            "click",
            pd.DataFrame(
                {
                    "user_id": ["a", "a", "b", "c", "c", "c"],
                    "item_id": ["x", "y", "z", "x", "y", "z"],
                    "count_id": np.arange(1, 7),
                }
            ),
            entities=["user", "item"],
            missing="insert",
        )


def test_add_interactions_table():
    dsb = DatasetBuilder()

    dsb.add_interactions(
        "click",
        pa.table(
            {
                "user_id": ["a", "a", "b", "c", "c", "c"],
                "item_id": ["x", "y", "z", "x", "y", "z"],
            }
        ),
        entities=["user", "item"],
        missing="insert",
    )

    ecs = dsb.entity_classes()
    assert set(ecs.keys()) == {"user", "item"}

    rcs = dsb.relationship_classes()
    assert set(rcs.keys()) == {"click"}

    ucls = ecs["user"]
    assert ucls.id_type == "str"

    ds = dsb.build()
    assert ds.user_count == 3
    assert ds.item_count == 3
    assert np.all(ds.users.ids() == ["a", "b", "c"])
    assert np.all(ds.items.ids() == ["x", "y", "z"])

    istats = ds.item_stats()
    assert np.all(istats["user_count"] == 2)

    log = ds.interaction_table(format="pandas", original_ids=True)
    assert isinstance(log, pd.DataFrame)
    assert all(log.columns == ["user_id", "item_id"])
    assert len(log) == 6

    mat = ds.interaction_matrix(format="structure")
    assert mat.nnz == 6
    assert np.all(mat.rowptrs == [0, 2, 3, 6])


def test_add_interactions_count_method():
    dsb = DatasetBuilder()

    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c"],
                "item_id": ["x", "y", "z", "x", "y", "z"],
                "count": np.arange(1, 7),
            }
        ),
        entities=["user", "item"],
        missing="insert",
    )

    ds = dsb.build()
    log = ds.interactions().count()
    assert log == 21


def test_add_interactions_count_method_with_null():
    dsb = DatasetBuilder()

    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c"],
                "item_id": ["x", "y", "z", "x", "y", "z"],
                "count": [1, 2, None, 4, 5, 6],
            }
        ),
        entities=["user", "item"],
        missing="insert",
    )

    ds = dsb.build()
    log = ds.interactions().count()
    assert log == 19


def test_add_interactions_error_bad_ids():
    dsb = DatasetBuilder()
    dsb.add_entities("user", ["a", "b", "c"])
    dsb.add_entities("item", ["z", "x", "y"])

    # fail with missing entities
    with raises(DataError, match="unknown"):
        dsb.add_interactions(
            "click",
            pd.DataFrame(
                {
                    "user_id": ["a", "a", "b", "c", "c", "c"],
                    "item_id": ["x", "y", "z", "x", "y", "w"],
                }
            ),
            entities=["user", "item"],
            missing="error",
        )

    # but using the correct ones passes
    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c"],
                "item_id": ["x", "y", "z", "x", "y", "z"],
            }
        ),
        entities=["user", "item"],
        missing="error",
    )

    ds = dsb.build()
    assert ds.user_count == 3
    assert ds.item_count == 3
    assert np.all(ds.users.ids() == ["a", "b", "c"])
    assert np.all(ds.items.ids() == ["x", "y", "z"])

    log = ds.interaction_table(format="pandas", original_ids=True)
    assert isinstance(log, pd.DataFrame)
    assert all(log.columns == ["user_id", "item_id"])
    assert len(log) == 6

    mat = ds.interaction_matrix(format="structure")
    assert mat.nnz == 6
    assert np.all(mat.rowptrs == [0, 2, 3, 6])

    assert len(ds.user_row("a")) == 2
    assert len(ds.user_row("b")) == 1
    assert len(ds.user_row("c")) == 3


def test_add_interactions_filter_bad_ids():
    dsb = DatasetBuilder()
    dsb.add_entities("user", ["a", "b", "c"])
    dsb.add_entities("item", ["z", "x", "y"])

    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c"],
                "item_id": ["x", "y", "z", "x", "w", "z"],
            }
        ),
        entities=["user", "item"],
        missing="filter",
    )

    ds = dsb.build()
    assert ds.user_count == 3
    assert ds.item_count == 3
    assert np.all(ds.users.ids() == ["a", "b", "c"])
    assert np.all(ds.items.ids() == ["x", "y", "z"])

    log = ds.interaction_table(format="pandas", original_ids=True)
    assert isinstance(log, pd.DataFrame)
    assert all(log.columns == ["user_id", "item_id"])
    assert len(log) == 5

    mat = ds.interaction_matrix(format="structure")
    assert mat.nnz == 5
    assert np.all(mat.rowptrs == [0, 2, 3, 5])

    assert len(ds.user_row("a")) == 2
    assert len(ds.user_row("b")) == 1
    assert len(ds.user_row("c")) == 2


def test_add_interactions_forbidden_repeat():
    dsb = DatasetBuilder()
    dsb.add_relationship_class("click", ["user", "item"], allow_repeats=False)
    with raises(DataError, match="repeated"):
        dsb.add_interactions(
            "click",
            pd.DataFrame(
                {
                    "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                    "item_id": ["x", "y", "z", "x", "y", "z", "z"],
                    "timestamp": np.arange(1, 8) * 10,
                }
            ),
            missing="insert",
        )
        dsb.build()


def test_add_multiple_interactions_forbidden_repeat():
    dsb = DatasetBuilder()
    dsb.add_relationship_class("click", ["user", "item"], allow_repeats=False)
    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c"],
                "item_id": ["x", "y", "z", "x"],
                "timestamp": np.arange(1, 5) * 10,
            }
        ),
        missing="insert",
    )
    with raises(DataError, match="repeated"):
        dsb.add_interactions(
            "click",
            pd.DataFrame(
                {
                    "user_id": ["c", "c", "b"],
                    "item_id": ["y", "z", "z"],
                    "timestamp": np.arange(5, 8) * 10,
                }
            ),
            missing="insert",
        )
        dsb.build()


def test_add_auto_entities():
    dsb = DatasetBuilder()

    with warns(DataWarning, match="specified"):
        dsb.add_interactions(
            "click",
            pd.DataFrame(
                {
                    "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                    "item_id": ["x", "y", "z", "x", "y", "z", "z"],
                    "timestamp": np.arange(1, 8) * 10,
                }
            ),
            missing="insert",
        )

    dsb.build()
    rsc = dsb.relationship_classes()["click"]
    assert rsc.repeats == AllowableTroolean.PRESENT
    assert rsc.entity_class_names == ["user", "item"]


def test_add_ratings(ml_ratings: pd.DataFrame):
    dsb = DatasetBuilder()
    dsb.add_interactions("rating", ml_ratings, entities=["user", "item"], missing="insert")

    rsc = dsb.relationship_classes()["rating"]
    assert rsc.entity_class_names == ["user", "item"]

    db = dsb.build()
    assert db.user_count == ml_ratings["user_id"].nunique()
    assert db.item_count == ml_ratings["item_id"].nunique()
    assert db.interaction_count == len(ml_ratings)

    ldf = db.interaction_table(format="pandas", original_ids=True)
    assert "rating" in ldf.columns
    assert ldf["rating"].mean() == approx(ml_ratings["rating"].mean())
    assert ldf["timestamp"].max() == ml_ratings["timestamp"].max()


def test_add_ratings_batched(ml_ratings: pd.DataFrame):
    dsb = DatasetBuilder()

    for bstart in range(0, len(ml_ratings), 1000):
        bend = min(len(ml_ratings), bstart + 1000)
        df = ml_ratings.iloc[bstart:bend]
        dsb.add_interactions("rating", df, entities=["user", "item"], missing="insert")

    rsc = dsb.relationship_classes()["rating"]
    assert rsc.entity_class_names == ["user", "item"]

    db = dsb.build()
    assert db.user_count == ml_ratings["user_id"].nunique()
    assert db.item_count == ml_ratings["item_id"].nunique()
    assert db.interaction_count == len(ml_ratings)

    ldf = db.interaction_table(format="pandas", original_ids=True)
    assert "rating" in ldf.columns
    assert ldf["rating"].mean() == approx(ml_ratings["rating"].mean())
    assert ldf["timestamp"].max() == ml_ratings["timestamp"].max()
