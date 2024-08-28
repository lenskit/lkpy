# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from uuid import UUID

import numpy as np
from typing_extensions import assert_type

from pytest import fail, raises, warns

from lenskit.pipeline import InputNode, Node, Pipeline, PipelineError
from lenskit.pipeline.types import TypecheckWarning


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


def test_lookup_optional():
    "lookup a node without failing"
    pipe = Pipeline()
    pipe.create_input("user", int, str)

    assert pipe.node("item", missing="none") is None


def test_lookup_missing():
    "lookup a node without failing"
    pipe = Pipeline()
    pipe.create_input("user", int, str)

    with raises(KeyError):
        pipe.node("item")


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


def test_single_input_required():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    def incr(msg: str) -> str:
        return msg

    node = pipe.add_component("return", incr, msg=msg)

    with raises(PipelineError, match="not specified"):
        pipe.run(node)


def test_single_optional_input():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str, None)

    def fill(msg: str | None) -> str:
        return msg if msg is not None else "undefined"

    node = pipe.add_component("return", fill, msg=msg)

    assert pipe.run(node) == "undefined"
    assert pipe.run(node, msg="hello") == "hello"


def test_single_input_typecheck():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    def incr(msg: str) -> str:
        return msg

    node = pipe.add_component("return", incr, msg=msg)

    with raises(TypeError):
        pipe.run(node, msg=47)


def test_component_type_mismatch():
    pipe = Pipeline()

    def incr(msg: str) -> str:
        return msg

    node = pipe.add_component("return", incr, msg=47)
    with raises(TypeError):
        pipe.run(node)


def test_component_unwired_input():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    def ident(msg: str, m2: str | None) -> str:
        if m2:
            return msg + m2
        else:
            return msg

    node = pipe.add_component("return", ident, msg=msg)
    assert pipe.run(node, msg="hello") == "hello"


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


def test_cycle():
    pipe = Pipeline()
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double)
    na = pipe.add_component("add", add, x=nd, y=b)
    pipe.connect(nd, x=na)

    with raises(PipelineError, match="cycle"):
        pipe.run(a=1, b=7)


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

    # run through the end
    assert pipe.run(a=1, b=7) == 10
    assert pipe.run(na, a=3, b=7) == 16
    # run only the first component
    assert pipe.run(nt, a=3, b=7) == 9

    # old node should be missing!
    with raises(PipelineError, match="not in pipeline"):
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


def test_run_all():
    pipe = Pipeline("test", "7.2")
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)

    pipe.alias("result", na)

    state = pipe.run_all(a=1, b=7)
    assert state["double"] == 2
    assert state["add"] == 9
    assert state["result"] == 9

    assert state.meta is not None
    assert state.meta.name == "test"
    assert state.meta.version == "7.2"
    assert state.meta.hash == pipe.config_hash()


def test_run_all_limit():
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

    state = pipe.run_all("double", a=1, b=7)
    assert state["double"] == 2
    assert "add" not in state
    assert "result" not in state


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

    with raises(PipelineError, match=r"input.*not specified"):
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

    # 3 * 2 + -3 = 3
    assert pipe.run(na, a=3) == 3


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


def test_fallback_fail_with_missing_options():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def negative(x: int) -> int | None:
        return None

    def double(x: int) -> int:
        return x * 2

    def add(x: int, y: int) -> int:
        return x + y

    nd = pipe.add_component("double", double, x=a)
    nn = pipe.add_component("negate", negative, x=a)
    fb = pipe.use_first_of("fill-operand", b, nn)
    na = pipe.add_component("add", add, x=nd, y=fb)

    with raises(RuntimeError, match="no alternative"):
        pipe.run(na, a=3)


def test_fallback_transitive():
    "test that a fallback works if a dependency's dependency fails"
    pipe = Pipeline()
    ia = pipe.create_input("a", int)
    ib = pipe.create_input("b", int)

    def double(x: int) -> int:
        return 2 * x

    # two components, each with a different input
    c1 = pipe.add_component("double-a", double, x=ia)
    c2 = pipe.add_component("double-b", double, x=ib)
    # use the first that succeeds
    c = pipe.use_first_of("result", c1, c2)

    # omitting the first input should result in the second component
    assert pipe.run(c, b=17) == 34


def test_fallback_transitive_deeper():
    "deeper transitive fallback test"
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def negative(x: int) -> int:
        return -x

    def double(x: int) -> int:
        return x * 2

    nd = pipe.add_component("double", double, x=a)
    nn = pipe.add_component("negate", negative, x=nd)
    nr = pipe.use_first_of("fill-operand", nn, b)

    assert pipe.run(nr, b=8) == 8


def test_fallback_transitive_nodefail():
    "deeper transitive fallback test"
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def negative(x: int) -> int | None:
        # make this return None in some cases to trigger failure
        if x >= 0:
            return -x
        else:
            return None

    def double(x: int) -> int:
        return x * 2

    nd = pipe.add_component("double", double, x=a)
    nn = pipe.add_component("negate", negative, x=nd)
    nr = pipe.use_first_of("fill-operand", nn, b)

    assert pipe.run(nr, a=2, b=8) == -4
    assert pipe.run(nr, a=-7, b=8) == 8


def test_pipeline_component_default():
    """
    Test that the last *component* is last.  It also exercises the warning logic
    for missing component types.
    """
    pipe = Pipeline()
    a = pipe.create_input("a", int)

    def add(x, y):  # type: ignore
        return x + y  # type: ignore

    with warns(TypecheckWarning):
        pipe.add_component("add", add, x=np.arange(10), y=a)  # type: ignore

    # the component runs
    assert np.all(pipe.run("add", a=5) == np.arange(5, 15))

    # the component is the default
    assert np.all(pipe.run(a=5) == np.arange(5, 15))
