from types import NoneType

from typing_extensions import assert_type

from lenskit.pipeline import InputNode, Node, Pipeline
from lenskit.pipeline.components import AutoConfig
from lenskit.pipeline.nodes import ComponentNode


class Prefixer(AutoConfig):
    prefix: str

    def __init__(self, prefix: str = "hello"):
        self.prefix = prefix

    def __call__(self, msg: str) -> str:
        return self.prefix + msg


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


def msg_ident(msg: str) -> str:
    return msg


def test_config_single_node():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    pipe.add_component("return", msg_ident, msg=msg)

    cfg = pipe.get_config()
    assert len(cfg.inputs) == 1
    assert len(cfg.components) == 1

    assert cfg.components["return"].code == "lenskit.tests.pipeline.test_save_load:msg_ident"
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

    pfx = Prefixer("scroll named ")
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
