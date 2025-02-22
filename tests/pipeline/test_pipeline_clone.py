# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from dataclasses import dataclass
from typing import Literal

from pytest import mark

from lenskit.pipeline import Component, Pipeline, PipelineBuilder
from lenskit.pipeline.nodes import ComponentInstanceNode


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


def _clone(builder: PipelineBuilder, pipe: Pipeline, what: Literal["pipe", "builder"]) -> Pipeline:
    match what:
        case "pipe":
            return pipe.clone()
        case "builder":
            return builder.clone().build()


@mark.parametrize("what", ["pipe", "builder"])
def test_pipeline_clone(what: Literal["pipe", "builder"]):
    builder = PipelineBuilder()
    msg = builder.create_input("msg", str)
    builder.add_component("prefix", Prefixer, PrefixConfig(prefix="scroll named "), msg=msg)
    builder.default_component("prefix")

    pipe = builder.build()
    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH"
    comp = pipe.node("prefix").component  # type: ignore

    p2 = _clone(builder, pipe, what)
    n2 = p2.node("prefix")
    assert isinstance(n2, ComponentInstanceNode)
    assert isinstance(n2.component, Prefixer)
    assert n2.component is not comp
    assert n2.component.config.prefix == comp.config.prefix  # type: ignore

    assert p2.run(msg="HACKEM MUCHE") == "scroll named HACKEM MUCHE"


@mark.parametrize("what", ["pipe", "builder"])
def test_pipeline_clone_with_function(what: Literal["pipe", "builder"]):
    comp = Prefixer(prefix="scroll named ")

    builder = PipelineBuilder()
    msg = builder.create_input("msg", str)
    pfx = builder.add_component("prefix", comp, msg=msg)
    builder.add_component("exclaim", exclaim, msg=pfx)
    builder.default_component("exclaim")

    pipe = builder.build()
    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH!"

    p2 = _clone(builder, pipe, what)

    assert p2.run(msg="HACKEM MUCHE") == "scroll named HACKEM MUCHE!"


@mark.parametrize("what", ["pipe", "builder"])
def test_pipeline_clone_with_nonconfig_class(what: Literal["pipe", "builder"]):
    comp = Prefixer(prefix="scroll named ")

    builder = PipelineBuilder()
    msg = builder.create_input("msg", str)
    pfx = builder.add_component("prefix", comp, msg=msg)
    builder.add_component("question", Question(), msg=pfx)
    builder.default_component("question")

    pipe = builder.build()
    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH?"

    p2 = _clone(builder, pipe, what)

    assert p2.run(msg="HACKEM MUCHE") == "scroll named HACKEM MUCHE?"


@mark.parametrize("what", ["pipe", "builder"])
def test_clone_defaults(what: Literal["pipe", "builder"]):
    builder = PipelineBuilder()
    msg = builder.create_input("msg", str)
    builder.default_connection("msg", msg)
    builder.add_component("return", exclaim)
    builder.default_component("return")

    pipe = builder.build()
    assert pipe.run(msg="hello") == "hello!"

    p2 = _clone(builder, pipe, what)

    assert p2.run(msg="hello") == "hello!"


@mark.parametrize("what", ["pipe", "builder"])
def test_clone_alias(what: Literal["pipe", "builder"]):
    builder = PipelineBuilder()
    msg = builder.create_input("msg", str)
    excl = builder.add_component("exclaim", exclaim, msg=msg)
    builder.alias("return", excl)

    pipe = builder.build()
    assert pipe.run("return", msg="hello") == "hello!"

    p2 = _clone(builder, pipe, what)

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
