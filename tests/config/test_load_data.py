# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from pydantic import BaseModel

from pytest import mark

from lenskit.config import load_config_data

TEST_DIR = Path(__file__).parent


class HelloModel(BaseModel):
    subject: str


@mark.parametrize("ext", ["json", "toml", "yml", "yaml"])
def test_load_raw(ext):
    file = TEST_DIR / f"hello.{ext}"

    data = load_config_data(file)
    assert isinstance(data, dict)
    assert data["subject"] == "world"


@mark.parametrize("ext", ["json", "toml", "yml", "yaml"])
def test_load_model(ext):
    file = TEST_DIR / f"hello.{ext}"

    data = load_config_data(file, HelloModel)
    assert isinstance(data, HelloModel)
    assert data.subject == "world"
