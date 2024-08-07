from types import NoneType

from typing_extensions import assert_type

from lenskit.pipeline import InputNode, Node, Pipeline


def test_serialize_input():
    "serialize with one input node"
    pipe = Pipeline()
    pipe.create_input("user", int, str)

    cfg = pipe.get_config()
    print(cfg)
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
