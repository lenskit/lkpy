# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import json
from os import fspath
from pathlib import Path

from click.testing import CliRunner

from lenskit.cli import lenskit
from lenskit.data import Dataset, ItemListCollection


def test_doctor():
    runner = CliRunner()
    result = runner.invoke(lenskit, ["doctor"])

    assert result.exit_code == 0


def test_doctor_full():
    runner = CliRunner()
    result = runner.invoke(lenskit, ["doctor", "--packages", "--paths"])

    assert result.exit_code == 0
