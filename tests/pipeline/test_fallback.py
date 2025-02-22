# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pytest import fail, raises

from lenskit.diagnostics import PipelineError
from lenskit.pipeline import PipelineBuilder


def test_fallback_input():
    pipe = PipelineBuilder()
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

    pipe = pipe.build()
    # 3 * 2 + -3 = 3
    assert pipe.run(na, a=3) == 3


def test_fallback_only_run_if_needed():
    pipe = PipelineBuilder()
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

    pipe = pipe.build()
    assert pipe.run(na, a=3, b=8) == 14


def test_fallback_fail_with_missing_options():
    pipe = PipelineBuilder()
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

    pipe = pipe.build()
    with raises(PipelineError, match="no data available"):
        pipe.run(na, a=3)


def test_fallback_transitive():
    "test that a fallback works if a dependency's dependency fails"
    pipe = PipelineBuilder()
    ia = pipe.create_input("a", int)
    ib = pipe.create_input("b", int)

    def double(x: int) -> int:
        return 2 * x

    # two components, each with a different input
    c1 = pipe.add_component("double-a", double, x=ia)
    c2 = pipe.add_component("double-b", double, x=ib)
    # use the first that succeeds
    c = pipe.use_first_of("result", c1, c2)

    pipe = pipe.build()
    # omitting the first input should result in the second component
    assert pipe.run(c, b=17) == 34


def test_fallback_transitive_deeper():
    "deeper transitive fallback test"
    pipe = PipelineBuilder()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    def negative(x: int) -> int:
        return -x

    def double(x: int) -> int:
        return x * 2

    nd = pipe.add_component("double", double, x=a)
    nn = pipe.add_component("negate", negative, x=nd)
    nr = pipe.use_first_of("fill-operand", nn, b)

    pipe = pipe.build()
    assert pipe.run(nr, b=8) == 8


def test_fallback_transitive_nodefail():
    "deeper transitive fallback test"
    pipe = PipelineBuilder()
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

    pipe = pipe.build()
    assert pipe.run(nr, a=2, b=8) == -4
    assert pipe.run(nr, a=-7, b=8) == 8
