# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pytest import raises

from lenskit.pipeline import PipelineBuilder
from lenskit.pipeline.types import Lazy


def proc_hello(msg: str) -> str:
    return f"Hello, {msg}"


def proc_incr(x: int) -> int:
    return x + 1


def proc_exclaim(msg: str) -> str:
    return msg + "!"


def proc_dft(msg: str = "unspecified") -> str:
    return f"msg:{msg}"


def lazy_exclaim(msg: Lazy[str], other: Lazy[int]) -> str:
    return msg.get() + "!"


def test_raise_invalid_input():
    build = PipelineBuilder()
    msg = build.create_input("message", str)
    hello = build.add_component("hello", proc_hello, msg=msg)

    pipe = build.build()

    out = pipe.run(hello, message="world")
    assert out == "Hello, world"

    with raises(TypeError, match="invalid data for input"):
        pipe.run(hello, message=5)


def test_raise_invalid_wired_input():
    build = PipelineBuilder()
    x = build.create_input("x", int)
    incr = build.add_component("incr", proc_incr, x=x)
    excl = build.add_component("exclaim", proc_exclaim, msg=incr)

    pipe = build.build()

    with raises(TypeError, match="found.*, expected"):
        pipe.run(excl, x=5)


def test_raise_lazy_input():
    build = PipelineBuilder()
    x = build.create_input("x", int)
    incr = build.add_component("incr", proc_incr, x=x)
    excl = build.add_component("exclaim", lazy_exclaim, msg=incr, other="bob")

    pipe = build.build()

    with raises(TypeError, match="found.*, expected"):
        pipe.run(excl, x=5)


def test_accept_default_input():
    build = PipelineBuilder()
    msg = build.create_input("msg", str)
    dft = build.add_component("label", proc_dft, msg=msg)

    pipe = build.build()
    out = pipe.run(dft, msg="FOOBIE BLETCH")
    assert out == "msg:FOOBIE BLETCH"


def test_accept_default_no_input():
    build = PipelineBuilder()
    msg = build.create_input("msg", str)
    dft = build.add_component("label", proc_dft, msg=msg)

    pipe = build.build()
    out = pipe.run(dft)
    assert out == "msg:unspecified"


def test_accept_default_unwired_input():
    build = PipelineBuilder()
    dft = build.add_component("label", proc_dft)

    pipe = build.build()
    out = pipe.run(dft)
    assert out == "msg:unspecified"


def test_reject_default_bad_input():
    build = PipelineBuilder()
    msg = build.create_input("msg", str)
    dft = build.add_component("label", proc_dft, msg=msg)

    pipe = build.build()
    with raises(TypeError, match="invalid data for input"):
        pipe.run(dft, msg=5)
