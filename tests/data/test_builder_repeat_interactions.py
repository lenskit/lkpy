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
from lenskit.data.schema import RepeatPolicy
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

    log = ds.interactions().pandas()
    assert isinstance(log, pd.DataFrame)
    assert all(log.columns == ["user_num", "item_num"])
    assert len(log) == 7


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


def test_repeated_interactions_count():  # add test with null counts
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
