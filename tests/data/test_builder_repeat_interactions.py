# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
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


def test_add_repeated_interactions():
    dsb = DatasetBuilder()

    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                "item_id": ["x", "y", "z", "x", "y", "z", "z"],
            }
        ),
        entities=["user", "item"],
        missing="insert",
    )

    ds = dsb.build()
    assert ds.user_count == 3
    assert ds.item_count == 3
    assert np.all(ds.users.ids() == ["a", "b", "c"])
    assert np.all(ds.items.ids() == ["x", "y", "z"])

    log = ds.interactions()
    assert log.schema.repeats == AllowableTroolean.PRESENT
    log = ds.interactions().pandas()
    assert isinstance(log, pd.DataFrame)
    assert all(log.columns == ["user_num", "item_num"])
    assert len(log) == 7


def test_remove_repeat_in_repeated_interactions():
    dsb = DatasetBuilder()
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
        allow_repeats=True,
        remove_repeats=True,
    )

    ds = dsb.build()
    log = ds.interactions()
    assert log.schema.repeats.is_allowed
    log = ds.interactions().pandas()
    assert isinstance(log, pd.DataFrame)
    assert all(log.columns == ["user_num", "item_num", "timestamp"])
    assert len(log) == 6
    assert len(log[log.user_num == 1]) == 1


def test_remove_repeat_with_forbidden_repeat():
    dsb = DatasetBuilder()
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
        allow_repeats=False,
        remove_repeats=True,
    )

    ds = dsb.build()
    log = ds.interactions()
    assert log.schema.repeats == AllowableTroolean.FORBIDDEN
    log = ds.interactions().pandas()
    assert isinstance(log, pd.DataFrame)
    assert all(log.columns == ["user_num", "item_num", "timestamp"])
    assert len(log) == 6
    assert len(log[log.user_num == 1]) == 1


def test_remove_duplicate_in_repeated_interactions():
    dsb = DatasetBuilder()
    dsb.add_relationship_class("click", ["user", "item"], allow_repeats=True, interaction=True)
    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                "item_id": ["x", "y", "z", "x", "y", "y", "z"],
                "tag_id": ["j", "i", "k", "j", "k", "i", "k"],
            }
        ),
        missing="insert",
        remove_repeats="exact",
    )

    ds = dsb.build()
    log = ds.interactions().pandas()
    assert isinstance(log, pd.DataFrame)
    assert all(log.columns == ["user_num", "item_num", "tag_id"])
    assert len(log) == 6
    assert len(log[log.user_num == 1]) == 1
    assert len(log[log.user_num == 2]) == 3


def test_remove_repeat_interaction_at_builder_saves_last_interaction():
    dsb = DatasetBuilder()
    dsb.add_relationship_class("click", ["user", "item"], allow_repeats=True, interaction=True)
    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                "item_id": ["x", "y", "z", "x", "y", "z", "z"],
                "timestamp": np.arange(1, 8) * 10,
                "rating": [3, 4, 5, 4, 3, 2, 3],
            }
        ),
        entities=["user", "item"],
        missing="insert",
        remove_repeats=True,
    )

    ds = dsb.build()
    log = ds.interactions().pandas()
    log = log[(log["user_num"] == 1) & (log["item_num"] == 2)]
    assert len(log) == 1
    assert log["rating"][5] == 3


def test_remove_repeat_interaction_at_matrix_saves_last_interaction():
    dsb = DatasetBuilder()
    dsb.add_relationship_class("click", ["user", "item"], allow_repeats=True, interaction=True)
    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                "item_id": ["x", "y", "z", "x", "y", "z", "z"],
                "timestamp": np.arange(1, 8) * 10,
                "rating": [3, 4, 5, 4, 3, 2, 3],
            }
        ),
        entities=["user", "item"],
        missing="insert",
        allow_repeats=True,
    )

    ds = dsb.build()
    log = ds.interactions().matrix().pandas()
    log = log[(log["user_num"] == 1) & (log["item_num"] == 2)]
    assert len(log) == 1
    assert log["rating"][2] == 3


def test_bad_interaction_matrix_call():
    dsb = DatasetBuilder()
    with raises(ValueError):
        dsb.add_interactions(
            "click",
            pd.DataFrame(
                {
                    "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                    "item_id": ["x", "y", "z", "x", "y", "z", "z"],
                }
            ),
            entities=["user", "item"],
            missing="insert",
        )
        ds = dsb.build()
        ds.interactions().matrix(row_entity="user", col_entity="user")


def test_repeated_interactions_timestamp():
    dsb = DatasetBuilder()

    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                "item_id": ["x", "y", "z", "x", "y", "z", "z"],
                "timestamp": np.arange(1, 8) * 10,
            }
        ),
        entities=["user", "item"],
        missing="insert",
    )

    ds = dsb.build()

    log = ds.interactions().pandas()
    assert isinstance(log, pd.DataFrame)
    assert all(log.columns == ["user_num", "item_num", "timestamp"])
    assert len(log) == 7

    matrs = ds.interactions().matrix()
    mat = matrs.csr_structure()
    assert mat.nnz == 6
    assert np.all(mat.rowptrs == [0, 2, 3, 6])

    mat = matrs.scipy(attribute="timestamp")
    assert mat[1, 2] == 70

    mat = matrs.scipy(attribute="first_timestamp")
    assert mat[1, 2] == 30

    mat = matrs.scipy(attribute="count")
    assert mat[1, 2] == 2

    assert len(ds.user_row("a")) == 2
    assert len(ds.user_row("b")) == 1
    assert len(ds.user_row("c")) == 3


def test_repeated_interactions_count():
    dsb = DatasetBuilder()

    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                "item_id": ["x", "y", "z", "x", "y", "z", "z"],
                "count": np.arange(1, 8),
            }
        ),
        entities=["user", "item"],
        missing="insert",
    )

    ds = dsb.build()

    matrs = ds.interactions().matrix()
    mat = matrs.csr_structure()
    assert mat.nnz == 6
    assert np.all(mat.rowptrs == [0, 2, 3, 6])

    mat = matrs.scipy(attribute="count")
    assert mat[1, 2] == 10


def test_repeated_interactions_count_with_null():
    dsb = DatasetBuilder()

    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                "item_id": ["x", "y", "z", "x", "y", "z", "z"],
                "count": [1, 2, None, 4, 5, 6, 7],
            }
        ),
        entities=["user", "item"],
        missing="insert",
    )

    ds = dsb.build()

    matrs = ds.interactions().matrix()
    mat = matrs.csr_structure()
    assert mat.nnz == 6
    assert np.all(mat.rowptrs == [0, 2, 3, 6])

    mat = matrs.scipy(attribute="count")
    assert mat[1, 2] == 8


def test_add_three_entities_interactions():
    dsb = DatasetBuilder()

    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                "item_id": ["x", "y", "z", "x", "y", "z", "z"],
                "tag_id": ["j", "i", "j", "j", "k", "i", "k"],
            }
        ),
        entities=["user", "item", "tag"],
        missing="insert",
    )

    ds = dsb.build()
    rset = ds.interactions()

    matrs_user_item = rset.matrix(row_entity="user", col_entity="item")
    mat = matrs_user_item.csr_structure()
    assert mat.nnz == 6
    assert np.all(mat.rowptrs == [0, 2, 3, 6])

    mat = matrs_user_item.scipy(attribute="count")
    assert mat[1, 2] == 2

    matrs_item_tag = rset.matrix(row_entity="item", col_entity="tag")
    mat = matrs_item_tag.csr_structure()
    assert mat.nnz == 6
    assert np.all(mat.rowptrs == [0, 1, 3, 6])

    mat = matrs_item_tag.scipy(attribute="count")
    assert mat[0, 1] == 2

    matrs_user_tag = rset.matrix(row_entity="user", col_entity="tag")
    mat = matrs_user_tag.csr_structure()
    assert mat.nnz == 7
    assert np.all(mat.rowptrs == [0, 2, 4, 7])

    mat = matrs_user_tag.scipy(attribute="count")
    assert np.all(mat.data == 1)


def test_matrix_relationship_set_cache():
    dsb = DatasetBuilder()

    dsb.add_interactions(
        "click",
        pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "c", "c", "b"],
                "item_id": ["x", "y", "z", "x", "y", "z", "z"],
            }
        ),
        entities=["user", "item"],
        missing="insert",
    )

    ds = dsb.build()
    matrix_user_item_a = ds.interactions().matrix(row_entity="user", col_entity="item")
    matrix_user_item_b = ds.interactions().matrix(row_entity="user", col_entity="item")

    assert matrix_user_item_a is matrix_user_item_b

    matrix_item_user = ds.interactions().matrix(row_entity="item", col_entity="user")

    assert matrix_item_user is not matrix_user_item_a
