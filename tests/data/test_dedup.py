# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pytest import mark

from lenskit._accel import data as _data_accel
from lenskit.data import Dataset
from lenskit.logging import Stopwatch, get_logger
from lenskit.testing import ml_20m

_log = get_logger(__name__)


def test_exact_no_dups(ml_ds: Dataset):
    "deduplicate data by entire rows"
    table = ml_ds.interactions().arrow()
    timer = Stopwatch()

    dedup = _data_accel.TableDedup(table.schema)

    for batch in table.to_batches(max_chunksize=10240):
        assert dedup.count_duplicates(batch) == 0

    _log.info("deduplicated ML Small in %s", timer)


@mark.bigdata
def test_exact_no_dups_big(ml_20m: Dataset):
    "deduplicate data by entire rows"
    table = ml_20m.interactions().arrow()
    timer = Stopwatch()

    dedup = _data_accel.TableDedup(table.schema)

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

    _log.info("deduplicated ML 25M in %s", timer)
