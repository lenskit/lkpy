# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from dataclasses import dataclass

from lenskit.pipeline.components import Component, component_inputs, component_return_type


@dataclass
class XConfig:
    suffix: str = ""


class XComp(Component):
    config: XConfig

    def __call__(self, msg: str) -> str:
        return msg + self.config.suffix


class CallObj:
    def __call__(self, q: str) -> bytes:
        return q.encode()


def comp_with_dft(q: str = "hello") -> str:
    return q + " world"


def test_empty_input():
    def func() -> int:
        return 9

    inputs = component_inputs(func)
    assert not inputs


def test_single_function_input():
    def func(x: int) -> int:
        return 9 + x

    inputs = component_inputs(func)
    assert len(inputs) == 1
    assert inputs["x"].type is int


def test_component_class_input():
    inputs = component_inputs(XComp)
    assert len(inputs) == 1
    assert inputs["msg"].type is str
    assert not inputs["msg"].has_default


def test_component_object_input():
    inputs = component_inputs(XComp())
    assert len(inputs) == 1
    assert inputs["msg"].type is str
    assert not inputs["msg"].has_default


def test_component_unknown_input():
    def func(x) -> int:  # type: ignore
        return x + 5  # type: ignore

    inputs = component_inputs(func)  # type: ignore
    assert len(inputs) == 1
    assert inputs["x"].type is None
    assert not inputs["x"].has_default


def test_callable_object_input():
    inputs = component_inputs(CallObj())
    assert len(inputs) == 1
    assert inputs["q"].type is str
    assert not inputs["q"].has_default


def test_callable_class_input():
    inputs = component_inputs(CallObj)
    assert len(inputs) == 1
    assert inputs["q"].type is str
    assert not inputs["q"].has_default


def test_callable_default_input():
    inputs = component_inputs(comp_with_dft)
    assert len(inputs) == 1
    assert inputs["q"].type is str
    assert inputs["q"].has_default


def test_function_return():
    def func(x: int) -> int:
        return x + 5

    rt = component_return_type(func)
    assert rt is int


def test_class_return():
    rt = component_return_type(XComp)
    assert rt is str


def test_instance_return():
    rt = component_return_type(XComp())
    assert rt is str


def test_unknown_return():
    def func():
        pass

    rt = component_return_type(func)
    assert rt is None


def test_callable_object_return():
    rt = component_return_type(CallObj())
    assert rt is bytes


def test_callable_class_return():
    rt = component_return_type(CallObj)
    assert rt is bytes
