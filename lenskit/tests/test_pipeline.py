# pyright: strict
from typing import assert_type
from uuid import UUID
from lenskit.pipeline import Node, Pipeline

from pytest import raises


def test_init_empty():
    pipe = Pipeline()
    assert len(pipe.nodes) == 0


def test_create_input():
    "create an input node"
    pipe = Pipeline()
    src = pipe.create_input("user", int, str)
    assert_type(src, Node[int | str])
    assert isinstance(src, Node)
    assert src.name == "user"

    assert len(pipe.nodes) == 1
    assert pipe.node("user") is src


def test_dup_input_fails():
    "create an input node"
    pipe = Pipeline()
    pipe.create_input("user", int, str)

    with raises(ValueError, match="has node"):
        pipe.create_input("user", UUID)
