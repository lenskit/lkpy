# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from dataclasses import dataclass

from lenskit.pipeline import PipelineBuilder
from lenskit.pipeline.components import Component
from lenskit.pipeline.nodes import ComponentNode


@dataclass
class PrefixConfig:
    prefix: str


class Prefixer(Component[str]):
    config: PrefixConfig

    def __call__(self, msg: str) -> str:
        return self.config.prefix + msg


class Question:
    "test component that is not configurable but is a class"

    def __call__(self, msg: str) -> str:
        return msg + "?"


def exclaim(msg: str) -> str:
    return msg + "!"


def test_pipeline_clone():
    comp = Prefixer(PrefixConfig("scroll named "))

    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)
    pipe.add_component("prefix", comp, msg=msg)
    pipe.default_component("prefix")

    pipe = pipe.build()
    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH"

    p2 = pipe.clone()
    n2 = p2.node("prefix")
    assert isinstance(n2, ComponentNode)
    assert isinstance(n2.component, Prefixer)
    assert n2.component is not comp
    assert n2.component.config.prefix == comp.config.prefix

    assert p2.run(msg="HACKEM MUCHE") == "scroll named HACKEM MUCHE"


def test_pipeline_clone_with_function():
    comp = Prefixer(prefix="scroll named ")

    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)
    pfx = pipe.add_component("prefix", comp, msg=msg)
    pipe.add_component("exclaim", exclaim, msg=pfx)
    pipe.default_component("prefix")

    pipe = pipe.build()
    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH!"

    p2 = pipe.clone()

    assert p2.run(msg="HACKEM MUCHE") == "scroll named HACKEM MUCHE!"


def test_pipeline_clone_with_nonconfig_class():
    comp = Prefixer(prefix="scroll named ")

    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)
    pfx = pipe.add_component("prefix", comp, msg=msg)
    pipe.add_component("question", Question(), msg=pfx)
    pipe.default_component("prefix")

    pipe = pipe.build()
    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH?"

    p2 = pipe.clone()

    assert p2.run(msg="HACKEM MUCHE") == "scroll named HACKEM MUCHE?"


def test_clone_defaults():
    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)
    pipe.default_connection("msg", msg)
    pipe.add_component("return", exclaim)
    pipe.default_component("return")

    pipe = pipe.build()
    assert pipe.run(msg="hello") == "hello!"

    p2 = pipe.clone()

    assert p2.run(msg="hello") == "hello!"


def test_clone_alias():
    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)
    excl = pipe.add_component("exclaim", exclaim, msg=msg)
    pipe.alias("return", excl)

    pipe = pipe.build()
    assert pipe.run("return", msg="hello") == "hello!"

    p2 = pipe.clone()

    assert p2.run("return", msg="hello") == "hello!"


def test_clone_hash():
    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)
    pipe.default_connection("msg", msg)
    excl = pipe.add_component("exclaim", exclaim)
    pipe.alias("return", excl)

    pipe = pipe.build()
    assert pipe.run("return", msg="hello") == "hello!"

    p2 = pipe.clone()

    assert p2.run("return", msg="hello") == "hello!"
    assert p2.config_hash == pipe.config_hash
