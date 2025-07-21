# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

from pytest import mark, skip

from lenskit.config import locate_configuration_root

LK_ROOT = Path(__file__).parent.parent.parent


def test_locate_lktoml_cwd(tmp_path: Path):
    lkt = tmp_path / "lenskit.toml"
    lkt.touch()

    path = locate_configuration_root(cwd=tmp_path)
    assert path == tmp_path


def test_locate_lktoml_parent(tmp_path: Path):
    lkt = tmp_path / "lenskit.toml"
    lkt.touch()

    child = tmp_path / "foo" / "bar"
    child.mkdir(parents=True, exist_ok=True)

    path = locate_configuration_root(cwd=child)
    assert path == tmp_path


def test_stop_git(tmp_path: Path):
    lkt = tmp_path / "lenskit.toml"
    lkt.touch()

    (tmp_path / "foo" / ".git").mkdir(parents=True)

    child = tmp_path / "foo" / "bar"
    child.mkdir(parents=True)

    path = locate_configuration_root(cwd=child)
    assert path is None


def test_stop_pyproject(tmp_path: Path):
    lkt = tmp_path / "lenskit.toml"
    lkt.touch()

    child = tmp_path / "foo" / "bar"
    child.mkdir(parents=True)

    (tmp_path / "foo" / "pyproject.toml").touch()

    path = locate_configuration_root(cwd=child)
    assert path is None


def test_locate_none(tmp_path: Path):
    child = tmp_path / "foo" / "bar"
    child.mkdir(parents=True, exist_ok=True)

    path = locate_configuration_root(cwd=child)
    assert path is None
