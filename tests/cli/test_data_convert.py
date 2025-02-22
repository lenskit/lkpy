# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import json
from os import fspath
from pathlib import Path

from click.testing import CliRunner

from lenskit.cli import lenskit
from lenskit.testing import ml_test_dir


def test_data_convert(tmpdir: Path):
    out_path = tmpdir / "ml-data"
    schema_file = out_path / "schema.json"

    runner = CliRunner()
    result = runner.invoke(
        lenskit, ["data", "convert", "--movielens", fspath(ml_test_dir), fspath(out_path)]
    )

    assert result.exit_code == 0
    assert schema_file.exists()

    data = json.loads(schema_file.read_text("utf8"))
    assert data["name"] == "ml-latest-small"
