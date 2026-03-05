# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import json
from os import fspath
from pathlib import Path

from click.testing import CliRunner

from lenskit.cli import lenskit
from lenskit.data import Dataset, ItemListCollection


def test_data_split(tmpdir: Path, ml_ds: Dataset):
    ds_path = tmpdir / "ml-data"
    ml_ds.save(ds_path)

    train_path = tmpdir / "ml-data.train"
    test_path = tmpdir / "ml-data.test.parquet"

    runner = CliRunner()
    result = runner.invoke(
        lenskit,
        ["data", "split", "--fraction", "0.2", fspath(ds_path)],
    )

    assert result.exit_code == 0
    assert train_path.exists()
    assert test_path.exists()

    trds = Dataset.load(train_path)
    print(trds)
    items = ItemListCollection.load_parquet(test_path)
    test_n = sum(len(il) for il in items.lists())
    tgt_n = int(ml_ds.interaction_count * 0.2)
    assert tgt_n - 1 <= test_n <= tgt_n + 1

    assert trds.interaction_count + test_n == ml_ds.interaction_count


def test_data_split_time(tmpdir: Path, ml_ds: Dataset):
    ds_path = tmpdir / "ml-data"
    ml_ds.save(ds_path)

    train_path = tmpdir / "ml-data.train"
    test_path = tmpdir / "ml-data.test.parquet"

    runner = CliRunner()
    result = runner.invoke(
        lenskit,
        ["data", "split", "--date", "2015-01-01", fspath(ds_path)],
    )

    assert result.exit_code == 0
    assert train_path.exists()
    assert test_path.exists()

    trds = Dataset.load(train_path)
    print(trds)
    items = ItemListCollection.load_parquet(test_path)
    test_n = sum(len(il) for il in items.lists())

    assert trds.interaction_count + test_n == ml_ds.interaction_count


def test_data_out_dir(tmpdir: Path, ml_ds: Dataset):
    ds_path = tmpdir / "ml-data"
    ml_ds.save(ds_path)

    out_dir = tmpdir / "split"
    train_path = out_dir / "train"
    test_path = out_dir / "test.parquet"

    runner = CliRunner()
    result = runner.invoke(
        lenskit,
        ["data", "split", "--fraction", "0.2", "-d", fspath(out_dir), fspath(ds_path)],
    )

    assert result.exit_code == 0
    assert train_path.exists()
    assert test_path.exists()

    trds = Dataset.load(train_path)
    print(trds)
    items = ItemListCollection.load_parquet(test_path)
    test_n = sum(len(il) for il in items.lists())
    tgt_n = int(ml_ds.interaction_count * 0.2)
    assert tgt_n - 1 <= test_n <= tgt_n + 1

    assert trds.interaction_count + test_n == ml_ds.interaction_count
