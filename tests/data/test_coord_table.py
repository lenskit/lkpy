# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pyarrow as pa

from lenskit._accel import data
from lenskit.data.vocab import Vocabulary


def test_empty():
    tbl = data.CoordinateTable(5)
    assert len(tbl) == 0
    assert tbl.dimensions() == 5
    assert tbl.unique_count() == 0


def test_add_rb():
    tbl = data.CoordinateTable(2)
    batch = pa.record_batch(
        {
            "row_num": pa.array([0, 0, 1, 1, 1, 3, 3], pa.int32()),
            "col_num": pa.array([0, 1, 0, 2, 5, 1, 3], pa.int32()),
        }
    )

    n, nuq = tbl.extend(batch)
    assert n == 7
    assert nuq == 7

    assert tbl.dimensions() == 2
    assert len(tbl) == 7
    assert tbl.unique_count() == 7

    assert tbl.contains(0, 1)
    assert not tbl.contains(0, 5)

    assert tbl.find(1, 2) == 3
    assert tbl.find(1, 3) is None


def test_add_dupes():
    tbl = data.CoordinateTable(2)
    batch = pa.record_batch(
        {
            "row_num": pa.array([0, 0, 1, 1, 1, 3, 3, 1], pa.int32()),
            "col_num": pa.array([0, 1, 0, 2, 5, 1, 3, 2], pa.int32()),
        }
    )

    n, nuq = tbl.extend(batch)
    assert n == 8
    assert nuq == 7

    assert tbl.dimensions() == 2
    assert len(tbl) == 8
    assert tbl.unique_count() == 7

    assert tbl.contains(0, 1)
    assert tbl.contains(1, 2)

    assert tbl.find(1, 2) == 7
    assert tbl.find(1, 3) is None


def test_add_batches(ml_ratings: pd.DataFrame, rng: np.random.Generator):
    ml_tbl = pa.Table.from_pandas(ml_ratings, preserve_index=False)

    items = Vocabulary(ml_tbl.column("item_id"), reorder=True)
    users = Vocabulary(ml_tbl.column("user_id"), reorder=True)

    ui_tbl = pa.table(
        {
            "user_num": users.numbers(ml_tbl.column("user_id"), format="arrow"),
            "item_num": items.numbers(ml_tbl.column("item_id"), format="arrow"),
        }
    )

    tbl = data.CoordinateTable(2)
    n, nuq = tbl.extend(ui_tbl.to_batches(10000))

    assert n == ml_tbl.num_rows
    assert nuq == n

    assert tbl.dimensions() == 2
    assert len(tbl) == n
    assert tbl.unique_count() == n

    for pos in rng.choice(ml_tbl.num_rows, size=500):
        uno = tbl._get_coord(0, pos)
        assert uno == ui_tbl.column("user_num")[pos].as_py()

        ino = tbl._get_coord(1, pos)
        assert ino == ui_tbl.column("item_num")[pos].as_py()
