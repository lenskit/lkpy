# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from lenskit.config import LenskitSettings, lenskit_config, load_configuration


def test_default_config():
    cfg = lenskit_config()
    assert cfg is not None
    assert cfg.random.seed is None


def test_load_toml():
    cfg = load_configuration(cfg_dir=Path(__file__).parent, _set_global=False)  # type: ignore
    assert cfg is not None
    assert cfg.random.seed == 42
    assert set(cfg.machines.keys()) == {"local", "shared"}
