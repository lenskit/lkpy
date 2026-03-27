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


def test_data_subset(tmpdir: Path, ml_ds: Dataset):
    ds_path = tmpdir / "ml-data"
    ml_ds.save(ds_path)

    out_path = tmpdir / "ml.subset.parquet"

    runner = CliRunner()
    result = runner.invoke(
        lenskit,
        ["data", "subset", "--sample-rows=100", "--item-lists", fspath(ds_path), fspath(out_path)],
    )

    assert result.exit_code == 0
    assert out_path.exists()

    items = ItemListCollection.load_parquet(out_path)
    assert len(items) == 100
