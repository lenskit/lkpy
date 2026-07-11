# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.pipeline.config import PipelineInput
from lenskit.pipeline.nodes import InputNode


def test_untyped_input():
    node = InputNode("scroll")

    cfg = node.to_config()
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types is None


def test_input_with_type():
    node = InputNode("scroll", type=str)

    cfg = node.to_config()
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types == {"str"}


def test_input_with_none():
    node = InputNode("scroll", type=str | None)

    cfg = node.to_config()
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types == {"None", "str"}


def test_input_with_generic():
    node = InputNode("scroll", type=list[str])

    cfg = node.to_config()
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types == {"list"}
