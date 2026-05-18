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


def test_pipeline_expand_conffile():
    runner = CliRunner()
    result = runner.invoke(lenskit, ["pipeline", "expand", "-c", "pipelines/als-implicit.toml"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "components" in data


def test_pipeline_expand_class():
    runner = CliRunner()
    result = runner.invoke(lenskit, ["pipeline", "expand", "-C", "lenskit.basic.BiasScorer"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "components" in data


def test_pipeline_expand_fails_no_scorer():
    runner = CliRunner()
    result = runner.invoke(lenskit, ["pipeline", "expand"])

    assert result.exit_code != 0
