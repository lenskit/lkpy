# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import random
import uuid

import pyarrow as pa

from pytest import mark

from lenskit._accel import data as _data_accel
from lenskit.data import Dataset
from lenskit.logging import Stopwatch, get_logger
from lenskit.testing import ml_20m

_log = get_logger(__name__)


@mark.parametrize("how", ["hash", "full", "short-hash"])
def test_exact_no_dups(ml_ds: Dataset, how: str):
    "deduplicate data by entire rows"
    table = ml_ds.interactions().arrow()
    timer = Stopwatch()

    dedup = _data_accel.TableDedup(table.schema, row_compare=how)

    for batch in table.to_batches(max_chunksize=10240):
        assert dedup.count_duplicates(batch) == 0

    _log.info("deduplicated ML Small in %s", timer)


@mark.bigdata
@mark.parametrize("how", ["hash", "full", "short-hash"])
def test_exact_no_dups_big(ml_20m: Dataset, how: str):
    "deduplicate data by entire rows"
    table = ml_20m.interactions().arrow()
    timer = Stopwatch()

    dedup = _data_accel.TableDedup(table.schema, row_compare=how)

    for batch in table.to_batches(max_chunksize=100240):
        assert dedup.count_duplicates(batch) == 0

    _log.info("deduplicated ML 25M in %s", timer)


def test_cols_no_dups(ml_ds: Dataset):
    "deduplicate data by user/item numbers"
    table = ml_ds.interactions().arrow()
    timer = Stopwatch()

    dedup = _data_accel.TableDedup(table.schema, columns=["user_num", "item_num"])

    for batch in table.to_batches(max_chunksize=10240):
        assert dedup.count_duplicates(batch) == 0

    _log.info("deduplicated ML Small in %s", timer)


@mark.bigdata
def test_cols_no_dups_big(ml_20m: Dataset):
    "deduplicate data by user/item numbers"
    table = ml_20m.interactions().arrow()
    timer = Stopwatch()

    dedup = _data_accel.TableDedup(table.schema, columns=["user_num", "item_num"])

    for batch in table.to_batches(max_chunksize=100240):
        assert dedup.count_duplicates(batch) == 0

    assert dedup.unique_count == table.num_rows
    _log.info("deduplicated ML 25M in %s", timer)


def test_dedup_one_column(ml_ds: Dataset):
    "deduplicate by a single entity"
    table = ml_ds.entities("item").arrow()

    dedup = _data_accel.TableDedup(table.schema, columns=["item_id"])

    for batch in table.to_batches(max_chunksize=1000):
        assert dedup.count_duplicates(batch) == 0

    assert dedup.unique_count == table.num_rows


@mark.bigdata
def test_dedup_one_column_big(ml_20m: Dataset):
    "deduplicate by a single entity"
    table = ml_20m.entities("item").arrow()

    dedup = _data_accel.TableDedup(table.schema, columns=["item_id"])

    for batch in table.to_batches(max_chunksize=1000):
        assert dedup.count_duplicates(batch) == 0

    assert dedup.unique_count == table.num_rows


def test_dedup_one_column_ints(random: random.Random):
    "deduplicate by a single int entity"
    ids = [random.randint(0, 500000) for _ in range(1000)]
    dupes = random.choices(ids, k=50)
    ids.extend(dupes)
    random.shuffle(ids)

    table = pa.Table.from_pydict(
        {"item_id": pa.array(ids, type=pa.int32()), "item_pos": range(len(ids))}
    )

    dedup = _data_accel.TableDedup(table.schema, columns=["item_id"])
    t2 = pa.Table.from_batches(dedup.dedup_batch(b) for b in table.to_batches(max_chunksize=200))

    assert t2.num_rows == len(set(ids))


def test_dedup_one_column_string(random: random.Random):
    "deduplicate by a single string entity"
    uuids = [uuid.uuid4() for _ in range(1000)]
    dupes = random.choices(uuids, k=50)
    uuids.extend(dupes)
    random.shuffle(uuids)

    table = pa.Table.from_pydict({"item_id": uuids, "item_pos": range(len(uuids))})

    dedup = _data_accel.TableDedup(table.schema, columns=["item_id"])
    t2 = pa.Table.from_batches(dedup.dedup_batch(b) for b in table.to_batches(max_chunksize=200))

    assert t2.num_rows == 1000
