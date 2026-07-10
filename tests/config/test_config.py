# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from lenskit.config import LenskitSettings, configure, lenskit_config, reconfigure


def test_default_config():
    cfg = lenskit_config()
    assert cfg is not None
    assert cfg.random.seed is None


def test_load_toml():
    cfg = configure(cfg_dir=Path(__file__).parent, _set_global=False)  # type: ignore
    assert cfg is not None
    assert cfg.random.seed == 42
    assert set(cfg.machines.keys()) == {"local", "shared"}


def test_reconfigure():
    base = lenskit_config()

    with reconfigure(cfg_dir=Path(__file__).parent):
        n2 = lenskit_config()
        assert n2 is not base
        assert n2.random.seed == 42
        assert n2.random.seed != base.random.seed

    outside = lenskit_config()
    assert outside == base
