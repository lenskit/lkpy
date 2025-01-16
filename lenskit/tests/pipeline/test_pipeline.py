# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from uuid import UUID

import numpy as np
from typing_extensions import assert_type

from pytest import mark, raises, warns

from lenskit.pipeline import PipelineBuilder, PipelineError
from lenskit.pipeline.nodes import InputNode, Node
from lenskit.pipeline.types import TypecheckWarning


def test_init_empty():
    pipe = PipelineBuilder()
    assert len(pipe.nodes()) == 0


def test_create_input():
    "create an input node"
    pipe = PipelineBuilder()
    src = pipe.create_input("user", int, str)
    assert_type(src, Node[int | str])
    assert isinstance(src, InputNode)
    assert src.name == "user"
    assert src.types == set([int, str])

    assert len(pipe.nodes()) == 1
    assert pipe.node("user") is src


def test_lookup_optional():
    "lookup a node without failing"
    pipe = PipelineBuilder()
    pipe.create_input("user", int, str)

    assert pipe.node("item", missing="none") is None


def test_lookup_missing():
    "lookup a node without failing"
    pipe = PipelineBuilder()
    pipe.create_input("user", int, str)

    with raises(KeyError):
        pipe.node("item")


def test_dup_input_fails():
    "create an input node"
    pipe = PipelineBuilder()
    pipe.create_input("user", int, str)

    with raises(ValueError, match="has node"):
        pipe.create_input("user", UUID)


def test_dup_component_fails():
    "create an input node"
    pipe = PipelineBuilder()
    pipe.create_input("user", int, str)

    with raises(ValueError, match="has node"):
        pipe.add_component("user", lambda x: x)  # type: ignore


def test_dup_alias_fails():
    "create an input node"
    pipe = PipelineBuilder()
    n = pipe.create_input("user", int, str)

    with raises(ValueError, match="has node"):
        pipe.alias("user", n)  # type: ignore


def test_alias():
    "alias a node"
    pipe = PipelineBuilder()
    user = pipe.create_input("user", int, str)

    pipe.alias("person", user)

    assert pipe.node("person") is user

    # aliases conflict as well
    with raises(ValueError):
        pipe.create_input("person", bytes)


def test_component_type():
    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)

    def incr(msg: str) -> str:
        return msg

    node = pipe.add_component("return", incr, msg=msg)
    assert node.name == "return"
    assert node.types == set([str])


def test_single_input():
    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)

    def incr(msg: str) -> str:
        return msg

    node = pipe.add_component("return", incr, msg=msg)

    pipe = pipe.build()

    ret = pipe.run(node, msg="hello")
    assert ret == "hello"

    ret = pipe.run(node, msg="world")
    assert ret == "world"


def test_single_input_required():
    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)

    def incr(msg: str) -> str:
        return msg

    node = pipe.add_component("return", incr, msg=msg)

    pipe = pipe.build()
    with raises(PipelineError, match="not specified"):
        pipe.run(node)


def test_single_optional_input():
    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str, None)

    def fill(msg: str | None) -> str:
        return msg if msg is not None else "undefined"

    node = pipe.add_component("return", fill, msg=msg)

    pipe = pipe.build()
    assert pipe.run(node) == "undefined"
    assert pipe.run(node, msg="hello") == "hello"


def test_single_input_typecheck():
    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)

    def incr(msg: str) -> str:
        return msg

    node = pipe.add_component("return", incr, msg=msg)

    pipe = pipe.build()
    with raises(TypeError):
        pipe.run(node, msg=47)


def test_component_type_mismatch():
    pipe = PipelineBuilder()

    def incr(msg: str) -> str:
        return msg

    node = pipe.add_component("return", incr, msg=47)
    pipe = pipe.build()
    with raises(TypeError):
        pipe.run(node)


def test_component_unwired_input():
    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)

    def ident(msg: str, m2: str | None) -> str:
        if m2:
            return msg + m2
        else:
            return msg

    node = pipe.add_component("return", ident, msg=msg)
    pipe = pipe.build()
    assert pipe.run(node, msg="hello") == "hello"


def test_chain():
    pipe = PipelineBuilder()
    x = pipe.create_input("x", int)

    def incr(x: int) -> int:
        return x + 1

    def triple(x: int) -> int:
        return x * 3

    ni = pipe.add_component("incr", incr, x=x)
    nt = pipe.add_component("triple", triple, x=ni)
    pipe.default_component(nt)

    pipe = pipe.build()
    # run default pipe
    ret = pipe.run(x=1)
    assert ret == 6

    # run explicitly
    assert pipe.run(nt, x=2) == 9

    # run only first node
    assert pipe.run(ni, x=10) == 11


def test_simple_graph():
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)
    pipe.default_component("add")

    pipe = pipe.build()
    assert pipe.run(a=1, b=7) == 9
    assert pipe.run(na, a=3, b=7) == 13
    assert pipe.run(nd, a=3, b=7) == 6


@mark.xfail(reason="cycle detection not yet implemented")
def test_cycle():
    pipe = PipelineBuilder()
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double)
    na = pipe.add_component("add", add, x=nd, y=b)
    pipe.connect(nd, x=na)

    with raises(PipelineError, match="cycle"):
        pipe.build()


def test_replace_component():
    pipe = PipelineBuilder()
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
    pipe.default_component(na)

    nt = pipe.replace_component("double", triple, x=a)

    pipe = pipe.build()

    # run through the end
    assert pipe.run(a=1, b=7) == 10
    assert pipe.run(na, a=3, b=7) == 16
    # run only the first component
    assert pipe.run(nt, a=3, b=7) == 9

    # old node should be missing!
    with raises(PipelineError, match="not in pipeline"):
        pipe.run(nd, a=3, b=7)


def test_default_wiring():
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    pipe.default_connection("y", b)

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd)
    pipe.default_component(na)

    pipe = pipe.build()
    assert pipe.run(a=1, b=7) == 9
    assert pipe.run(na, a=3, b=7) == 13


def test_run_by_name():
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    pipe.add_component("add", add, x=nd, y=b)

    pipe = pipe.build()
    assert pipe.run("double", a=1, b=7) == 2


def test_run_tuple_name():
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    pipe.add_component("add", add, x=nd, y=b)

    pipe = pipe.build()
    res = pipe.run(("double",), a=1, b=7)
    assert isinstance(res, tuple)
    assert res[0] == 2


def test_run_tuple_pair():
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    pipe.add_component("add", add, x=nd, y=b)

    pipe = pipe.build()
    res = pipe.run(("double", "add"), a=1, b=7)
    assert isinstance(res, tuple)
    d, a = res
    assert d == 2
    assert a == 9


def test_invalid_type():
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    pipe.add_component("add", add, x=nd, y=b)

    pipe = pipe.build()
    with raises(TypeError):
        pipe.run("add", a=1, b="seven")


def test_run_by_alias():
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)

    pipe.alias("result", na)

    pipe = pipe.build()
    assert pipe.run("result", a=1, b=7) == 9


def test_run_all():
    pipe = PipelineBuilder("test", "7.2")
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)

    pipe.alias("result", na)

    pipe = pipe.build()
    state = pipe.run_all(a=1, b=7)
    assert state["double"] == 2
    assert state["add"] == 9
    assert state["result"] == 9

    assert state.meta is not None
    assert state.meta.name == "test"
    assert state.meta.version == "7.2"
    assert state.meta.hash == pipe.config_hash


def test_run_all_limit():
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)

    pipe.alias("result", na)

    pipe = pipe.build()
    state = pipe.run_all("double", a=1, b=7)
    assert state["double"] == 2
    assert "add" not in state
    assert "result" not in state


def test_connect_literal():
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=2)

    pipe = pipe.build()
    assert pipe.run(na, a=3) == 8


def test_connect_literal_explicit():
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=pipe.literal(2))

    pipe = pipe.build()
    assert pipe.run(na, a=3) == 8


def test_fail_missing_input():
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)

    pipe = pipe.build()

    with raises(PipelineError, match=r"input.*not specified"):
        pipe.run(na, a=3)

    # missing inputs only matter if they are required
    assert pipe.run(nd, a=3) == 6


def test_pipeline_component_default():
    """
    Test that the default component is run correctly.
    """
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)

    def add(x, y):  # type: ignore
        return x + y  # type: ignore

    with warns(TypecheckWarning):
        pipe.add_component("add", add, x=np.arange(10), y=a)  # type: ignore
    pipe.default_component("add")

    cfg = pipe.build_config()
    assert cfg.default == "add"

    pipe = pipe.build()
    # the component runs
    assert np.all(pipe.run("add", a=5) == np.arange(5, 15))

    # the component is the default
    assert np.all(pipe.run(a=5) == np.arange(5, 15))
