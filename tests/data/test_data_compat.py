# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import subprocess as sp
import sys
from os import environ, fspath
from pathlib import Path

from pytest import mark, skip

from lenskit.data.dataset import Dataset
from lenskit.data.matrix import SparseRowArray
from lenskit.logging import get_logger
from lenskit.testing import ml_test_dir

# The LensKit versions we want to test backwards compatibility with
LK_VERSIONS = ["2025.1.1"]

_log = get_logger(__name__)
_ml_path = Path("data/ml-20m.zip")


@mark.slow
@mark.parametrize("version", LK_VERSIONS)
def test_data_backwards_compat(version, tmpdir: Path):
    "Test that we can load datasets prepared by old versions."
    if version <= "2025.1.1" and sys.version_info >= (3, 14):
        skip("2025.1.1 doesn't run on Python 3.14")

    _log.info("processing ML file", version=version)

    try:
        sp.call(["uvx", "--version"])
    except FileNotFoundError:
        skip("uvx not installed")

    out_path = tmpdir / "ml-small.lk"
    pkg = f"lenskit=={version}"

    sp.check_call(
        [
            "uvx",
            "--isolated",
            pkg,
            "data",
            "convert",
            "--movielens",
            fspath(ml_test_dir),
            fspath(out_path),
        ],
        env=environ | {"UV_TORCH_BACKEND": "cpu"},
    )

    _log.info("loading dataset")
    ml = Dataset.load(out_path)
    assert ml.schema.version is not None


@mark.realdata
@mark.skipif(not _ml_path.exists(), reason="ml-20m not available")
@mark.parametrize("version", LK_VERSIONS)
def test_data_backwards_ml20m(version, tmpdir: Path):
    "Test that we can load datasets prepared by old versions (ML20M)."
    try:
        sp.call(["uvx", "--version"])
    except FileNotFoundError:
        skip("uvx not installed")

    _log.info("processing ML file", version=version)

    out_path = tmpdir / "ml-20m"
    pkg = f"lenskit=={version}"

    sp.check_call(
        [
            "uvx",
            "--isolated",
            pkg,
            "data",
            "convert",
            "--movielens",
            fspath(_ml_path),
            fspath(out_path),
        ],
        env=environ | {"UV_TORCH_BACKEND": "cpu"},
    )

    _log.info("loading dataset")
    ml = Dataset.load(out_path)
    assert ml.schema.version is not None

    arr = ml.entities("item").attribute("tag_counts").arrow()
    assert isinstance(arr, SparseRowArray)
