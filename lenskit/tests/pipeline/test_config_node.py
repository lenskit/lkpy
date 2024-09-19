# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.pipeline.config import PipelineInput
from lenskit.pipeline.nodes import InputNode


def test_untyped_input():
    node = InputNode("scroll")

    cfg = PipelineInput.from_node(node)
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types is None


def test_input_with_type():
    node = InputNode("scroll", types={str})

    cfg = PipelineInput.from_node(node)
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types == {"str"}


def test_input_with_none():
    node = InputNode("scroll", types={str, type(None)})

    cfg = PipelineInput.from_node(node)
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types == {"None", "str"}


def test_input_with_generic():
    node = InputNode("scroll", types={list[str]})

    cfg = PipelineInput.from_node(node)
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types == {"list"}
