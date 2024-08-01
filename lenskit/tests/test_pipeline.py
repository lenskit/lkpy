# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from typing import assert_type
from uuid import UUID

from pytest import fail, raises

from lenskit.pipeline import Node, Pipeline
from lenskit.pipeline._impl import InputNode


def test_init_empty():
    pipe = Pipeline()
    assert len(pipe.nodes) == 0


def test_create_input():
    "create an input node"
    pipe = Pipeline()
    src = pipe.create_input("user", int, str)
    assert_type(src, Node[int | str])
    assert isinstance(src, InputNode)
    assert src.name == "user"
    assert src.types == set([int, str])

    assert len(pipe.nodes) == 1
    assert pipe.node("user") is src


def test_dup_input_fails():
    "create an input node"
    pipe = Pipeline()
    pipe.create_input("user", int, str)

    with raises(ValueError, match="has node"):
        pipe.create_input("user", UUID)


def test_dup_component_fails():
    "create an input node"
    pipe = Pipeline()
    pipe.create_input("user", int, str)

    with raises(ValueError, match="has node"):
        pipe.add_component("user", lambda x: x)  # type: ignore


def test_dup_alias_fails():
    "create an input node"
    pipe = Pipeline()
    n = pipe.create_input("user", int, str)

    with raises(ValueError, match="has node"):
        pipe.alias("user", n)  # type: ignore


def test_alias():
    "alias a node"
    pipe = Pipeline()
    user = pipe.create_input("user", int, str)

    pipe.alias("person", user)

    assert pipe.node("person") is user

    # aliases conflict as well
    with raises(ValueError):
        pipe.create_input("person", bytes)


def test_component_type():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    def incr(msg: str) -> str:
        return msg

    node = pipe.add_component("return", incr, msg=msg)
    assert node.name == "return"
    assert node.types == set([str])


def test_single_input():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    def incr(msg: str) -> str:
        return msg

    node = pipe.add_component("return", incr, msg=msg)

    ret = pipe.run(node, msg="hello")
    assert ret == "hello"

    ret = pipe.run(node, msg="world")
    assert ret == "world"


def test_single_input_typecheck():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    def incr(msg: str) -> str:
        return msg

    node = pipe.add_component("return", incr, msg=msg)

    with raises(TypeError):
        pipe.run(node, msg=47)


def test_chain():
    pipe = Pipeline()
    x = pipe.create_input("x", int)

    def incr(x: int) -> int:
        return x + 1

    def triple(x: int) -> int:
        return x * 3

    ni = pipe.add_component("incr", incr, x=x)
    nt = pipe.add_component("triple", triple, x=ni)

    # run default pipe
    ret = pipe.run(x=1)
    assert ret == 6

    # run explicitly
    assert pipe.run(nt, x=2) == 9

    # run only first node
    assert pipe.run(ni, x=10) == 11


def test_simple_graph():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)

    assert pipe.run(a=1, b=7) == 9
    assert pipe.run(na, a=3, b=7) == 13
    assert pipe.run(nd, a=3, b=7) == 6


def test_replace_component():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def triple(x: int) -> int:
        return x * 3

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)

    nt = pipe.replace_component("double", triple, x=a)

    assert pipe.run(a=1, b=7) == 9
    assert pipe.run(na, a=3, b=7) == 13
    assert pipe.run(nt, a=3, b=7) == 9
    with raises(RuntimeError, match="not in pipeline"):
        pipe.run(nd, a=3, b=7)


def test_default_wiring():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    pipe.set_default("y", b)

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd)

    assert pipe.run(a=1, b=7) == 9
    assert pipe.run(na, a=3, b=7) == 13
    with raises(RuntimeError, match="not in pipeline"):
        pipe.run(nd, a=3, b=7)


def test_run_by_name():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    pipe.add_component("add", add, x=nd, y=b)

    assert pipe.run("double", a=1, b=7) == 2


def test_invalid_type():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    pipe.add_component("add", add, x=nd, y=b)

    with raises(TypeError):
        pipe.run(a=1, b="seven")


def test_run_by_alias():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)

    pipe.alias("result", na)

    assert pipe.run("result", a=1, b=7) == 9


def test_connect_literal():
    pipe = Pipeline()
    a = pipe.create_input("a", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=2)

    assert pipe.run(na, a=3) == 8


def test_connect_literal_explicit():
    pipe = Pipeline()
    a = pipe.create_input("a", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=pipe.literal(2))

    assert pipe.run(na, a=3) == 8


def test_fail_missing_input():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)

    with raises(RuntimeError, match=r"input.*not specified"):
        pipe.run(na, a=3)

    # missing inputs only matter if they are required
    assert pipe.run(nd, a=3) == 6


def test_fallback_input():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def negative(x: int) -> int:
        return -x

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    nn = pipe.add_component("negate", negative, x=a)
    fb = pipe.use_first_of("fill-operand", b, nn)
    na = pipe.add_component("add", add, x=nd, y=fb)

    assert pipe.run(na, a=3) == 0


def test_fallback_only_run_if_needed():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def negative(x: int) -> int:
        fail("fallback component run when not needed")

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    nn = pipe.add_component("negate", negative, x=a)
    fb = pipe.use_first_of("fill-operand", b, nn)
    na = pipe.add_component("add", add, x=nd, y=fb)

    assert pipe.run(na, a=3, b=8) == 14
