# pyright: strict
from dataclasses import dataclass

from lenskit.pipeline.components import Component, component_inputs, component_return_type


@dataclass
class TestConfig:
    suffix: str = ""


class TestComp(Component):
    config: TestConfig

    def __call__(self, msg: str) -> str:
        return msg + self.config.suffix


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
    assert inputs["x"] is int


def test_component_class_input():
    inputs = component_inputs(TestComp)
    assert len(inputs) == 1
    assert inputs["msg"] is str


def test_component_object_input():
    inputs = component_inputs(TestComp())
    assert len(inputs) == 1
    assert inputs["msg"] is str


def test_component_unknown_input():
    def func(x) -> int:  # type: ignore
        return x + 5  # type: ignore

    inputs = component_inputs(func)  # type: ignore
    assert len(inputs) == 1
    assert inputs["x"] is None


def test_function_return():
    def func(x: int) -> int:
        return x + 5

    rt = component_return_type(func)
    assert rt is int


def test_class_return():
    rt = component_return_type(TestComp)
    assert rt is str


def test_instance_return():
    rt = component_return_type(TestComp())
    assert rt is str


def test_unknown_return():
    def func():
        pass

    rt = component_return_type(func)
    assert rt is None
