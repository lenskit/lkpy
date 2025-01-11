# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import re
from dataclasses import dataclass
from types import NoneType

import numpy as np
from typing_extensions import assert_type

from pytest import fail, warns

from lenskit.pipeline import Pipeline, PipelineWarning
from lenskit.pipeline.components import Component
from lenskit.pipeline.config import PipelineConfig
from lenskit.pipeline.nodes import ComponentNode, InputNode

_log = logging.getLogger(__name__)


# region Test Components
@dataclass
class PrefixConfig:
    prefix: str


class Prefixer(Component[str]):
    config: PrefixConfig

    def __call__(self, msg: str) -> str:
        return self.config.prefix + msg


def negative(x: int) -> int:
    return -x


def double(x: int) -> int:
    return x * 2


def add(x: int | np.ndarray, y: int) -> int | np.ndarray:
    return x + y


def msg_ident(msg: str) -> str:
    return msg


def msg_prefix(prefix: str, msg: str) -> str:
    return prefix + msg


# endregion


def test_serialize_input():
    "serialize with one input node"
    pipe = Pipeline("test")
    pipe.create_input("user", int, str)

    cfg = pipe.get_config()
    print(cfg)
    assert cfg.meta.name == "test"
    assert len(cfg.inputs) == 1
    assert cfg.inputs[0].name == "user"
    assert cfg.inputs[0].types == {"int", "str"}


def test_round_trip_input():
    "serialize with one input node"
    pipe = Pipeline()
    pipe.create_input("user", int, str)

    cfg = pipe.get_config()
    print(cfg)

    p2 = Pipeline.from_config(cfg)
    i2 = p2.node("user")
    assert isinstance(i2, InputNode)
    assert i2.name == "user"
    assert i2.types == {int, str}


def test_round_trip_optional_input():
    "serialize with one input node"
    pipe = Pipeline()
    pipe.create_input("user", int, str, None)

    cfg = pipe.get_config()
    assert cfg.inputs[0].types == {"int", "str", "None"}

    p2 = Pipeline.from_config(cfg)
    i2 = p2.node("user")
    assert isinstance(i2, InputNode)
    assert i2.name == "user"
    assert i2.types == {int, str, NoneType}


def test_config_single_node():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    pipe.add_component("return", msg_ident, msg=msg)

    cfg = pipe.get_config()
    assert len(cfg.inputs) == 1
    assert len(cfg.components) == 1

    assert re.match(
        r"((lenskit\.)?tests\.)?pipeline\.test_save_load:msg_ident", cfg.components["return"].code
    )
    assert cfg.components["return"].config is None
    assert cfg.components["return"].inputs == {"msg": "msg"}


def test_round_trip_single_node():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    pipe.add_component("return", msg_ident, msg=msg)

    cfg = pipe.get_config()

    p2 = Pipeline.from_config(cfg)
    assert len(p2.nodes) == 2
    r2 = p2.node("return")
    assert isinstance(r2, ComponentNode)
    assert r2.component is msg_ident
    assert r2.connections == {"msg": "msg"}

    assert p2.run("return", msg="foo") == "foo"


def test_configurable_component():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    pfx = Prefixer(prefix="scroll named ")
    pipe.add_component("prefix", pfx, msg=msg)

    cfg = pipe.get_config()
    assert cfg.components["prefix"].config == {"prefix": "scroll named "}

    p2 = Pipeline.from_config(cfg)
    assert len(p2.nodes) == 2
    r2 = p2.node("prefix")
    assert isinstance(r2, ComponentNode)
    assert isinstance(r2.component, Prefixer)
    assert r2.component is not pfx
    assert r2.connections == {"msg": "msg"}

    assert p2.run("prefix", msg="HACKEM MUCHE") == "scroll named HACKEM MUCHE"

    print("hash:", pipe.config_hash())
    assert pipe.config_hash() is not None
    assert p2.config_hash() == pipe.config_hash()


def test_save_defaults():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)
    pipe.set_default("msg", msg)
    pipe.add_component("return", msg_ident)

    assert pipe.run(msg="hello") == "hello"

    cfg = pipe.get_config()
    p2 = Pipeline.from_config(cfg)

    assert p2.run(msg="hello") == "hello"


def test_hashes_different():
    p1 = Pipeline()
    p2 = Pipeline()

    a1 = p1.create_input("a", int)
    a2 = p2.create_input("a", int)

    # at this point the hashes should be the same
    _log.info("p1 stage 1 hash: %s", p1.config_hash())
    _log.info("p2 stage 1 hash: %s", p2.config_hash())
    assert p1.config_hash() == p2.config_hash()

    p1.add_component("proc", negative, x=a1)
    p2.add_component("proc", double, x=a2)

    # with different components, they should be different
    _log.info("p1 stage 2 hash: %s", p1.config_hash())
    _log.info("p2 stage 2 hash: %s", p2.config_hash())
    assert p1.config_hash() != p2.config_hash()


def test_save_with_fallback():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    nd = pipe.add_component("double", double, x=a)
    nn = pipe.add_component("negate", negative, x=a)
    fb = pipe.use_first_of("fill-operand", b, nn)
    pipe.add_component("add", add, x=nd, y=fb)

    cfg = pipe.get_config()
    json = cfg.model_dump_json(exclude_none=True)
    print(json)
    c2 = PipelineConfig.model_validate_json(json)

    p2 = Pipeline.from_config(c2)

    # 3 * 2 + -3 = 3
    assert p2.run("fill-operand", "add", a=3) == (-3, 3)


def test_hash_validate():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    pfx = Prefixer(prefix="scroll named ")
    pipe.add_component("prefix", pfx, msg=msg)

    cfg = pipe.get_config()
    print("initial config:", cfg.model_dump_json(indent=2))
    assert cfg.meta.hash is not None
    cfg.components["prefix"].config["prefix"] = "scroll called "  # type: ignore
    print("modified config:", cfg.model_dump_json(indent=2))

    with warns(PipelineWarning):
        Pipeline.from_config(cfg)


def test_alias_input():
    "just an input node and an alias"
    pipe = Pipeline()
    user = pipe.create_input("user", int, str)

    pipe.alias("person", user)

    cfg = pipe.get_config()

    p2 = Pipeline.from_config(cfg)
    assert p2.run("person", user=32) == 32


def test_alias_node():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)
    pipe.alias("result", na)

    assert pipe.run("result", a=5, b=7) == 17

    p2 = pipe.clone("pipeline-config")
    assert p2.run("result", a=5, b=7) == 17


def test_literal():
    pipe = Pipeline("literal-prefix")
    msg = pipe.create_input("msg", str)

    pipe.add_component("prefix", msg_prefix, prefix=pipe.literal("hello, "), msg=msg)

    assert pipe.run(msg="HACKEM MUCHE") == "hello, HACKEM MUCHE"

    print(pipe.get_config().model_dump_json(indent=2))
    p2 = pipe.clone("pipeline-config")
    assert p2.run(msg="FOOBIE BLETCH") == "hello, FOOBIE BLETCH"


def test_literal_array():
    pipe = Pipeline("literal-add-array")
    a = pipe.create_input("a", int)

    pipe.add_component("add", add, x=np.arange(10), y=a)

    res = pipe.run(a=5)
    assert np.all(res == np.arange(5, 15))

    print(pipe.get_config().model_dump_json(indent=2))
    p2 = pipe.clone("pipeline-config")
    assert np.all(p2.run(a=5) == np.arange(5, 15))


def test_stable_with_literals():
    "test that two identical pipelines have the same hash, even with literals"
    p1 = Pipeline("literal-add-array")
    a = p1.create_input("a", int)
    p1.add_component("add", add, x=np.arange(10), y=a)

    p2 = Pipeline("literal-add-array")
    a = p2.create_input("a", int)
    p2.add_component("add", add, x=np.arange(10), y=a)

    assert p1.config_hash() == p2.config_hash()
