# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from dataclasses import dataclass
from typing import Literal

from pytest import mark, raises

from lenskit.diagnostics import PipelineError
from lenskit.pipeline import Component, Pipeline, PipelineBuilder
from lenskit.pipeline.nodes import ComponentInstanceNode


@dataclass
class PrefixConfig:
    prefix: str


class Prefixer(Component[str]):
    config: PrefixConfig

    def __call__(self, msg: str) -> str:
        return self.config.prefix + msg


def lowercase(msg: str) -> str:
    return msg.lower()


def test_modify_pipeline_nomod():
    builder = PipelineBuilder()
    msg = builder.create_input("msg", str)
    builder.add_component("prefix", Prefixer, PrefixConfig(prefix="scroll named "), msg=msg)
    builder.default_component("prefix")

    pipe = builder.build()
    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH"
    comp = pipe.node("prefix").component  # type: ignore

    b2 = pipe.modify()
    p2 = b2.build()
    n2 = p2.node("prefix")
    assert isinstance(n2, ComponentInstanceNode)
    assert isinstance(n2.component, Prefixer)
    assert n2.component is comp

    assert p2.run(msg="HACKEM MUCHE") == "scroll named HACKEM MUCHE"


def test_modify_pipeline_replace():
    builder = PipelineBuilder()
    msg = builder.create_input("msg", str)
    builder.add_component("prefix", Prefixer, PrefixConfig(prefix="scroll named "), msg=msg)
    builder.default_component("prefix")

    pipe = builder.build()
    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH"
    comp = pipe.node("prefix").component  # type: ignore

    b2 = pipe.modify()
    b2.replace_component("prefix", Prefixer, PrefixConfig(prefix="scroll called "))

    p2 = b2.build()
    n2 = p2.node("prefix")
    assert isinstance(n2, ComponentInstanceNode)
    assert isinstance(n2.component, Prefixer)
    assert n2.component is not comp

    assert p2.run(msg="HACKEM MUCHE") == "scroll called HACKEM MUCHE"


def test_modify_pipeline_new_input():
    builder = PipelineBuilder()
    msg = builder.create_input("msg", str)
    builder.add_component("prefix", Prefixer, PrefixConfig(prefix="scroll named "), msg=msg)
    builder.default_component("prefix")

    pipe = builder.build()
    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH"
    comp = pipe.node("prefix").component  # type: ignore

    b2 = pipe.modify()
    up = b2.add_component("upper", lowercase, msg=msg)
    b2.replace_component("prefix", Prefixer, PrefixConfig(prefix="scroll called "), msg=up)

    p2 = b2.build()
    n2 = p2.node("prefix")
    assert isinstance(n2, ComponentInstanceNode)
    assert isinstance(n2.component, Prefixer)
    assert n2.component is not comp

    assert p2.run(msg="HACKEM MUCHE") == "scroll called hackem muche"


def test_modify_pipeline_clear_input():
    builder = PipelineBuilder()
    msg = builder.create_input("msg", str)
    builder.add_component("prefix", Prefixer, PrefixConfig(prefix="scroll named "), msg=msg)
    builder.default_component("prefix")

    pipe = builder.build()
    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH"
    comp = pipe.node("prefix").component  # type: ignore

    b2 = pipe.modify()
    b2.replace_component("prefix", Prefixer, PrefixConfig(prefix="scroll called "))
    b2.clear_inputs("prefix")

    p2 = b2.build()
    n2 = p2.node("prefix")
    assert isinstance(n2, ComponentInstanceNode)
    assert isinstance(n2.component, Prefixer)
    assert n2.component is not comp

    with raises(PipelineError):
        p2.run(msg="HACKEM MUCHE")
