from types import NoneType

from typing_extensions import assert_type

from pytest import fail, warns

from lenskit.pipeline import InputNode, Node, Pipeline, PipelineWarning
from lenskit.pipeline.components import AutoConfig
from lenskit.pipeline.config import PipelineConfig
from lenskit.pipeline.nodes import ComponentNode


class Prefixer(AutoConfig):
    prefix: str

    def __init__(self, prefix: str = "hello"):
        self.prefix = prefix

    def __call__(self, msg: str) -> str:
        return self.prefix + msg


def test_serialize_input():
    "serialize with one input node"
    pipe = Pipeline("test")
    pipe.create_input("user", int, str)

    cfg = pipe.get_config()
    print(cfg)
    assert cfg.meta.name == "test"
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

    print("hash:", pipe.config_hash())
    assert pipe.config_hash() is not None
    assert p2.config_hash() == pipe.config_hash()


def negative(x: int) -> int:
    return -x


def double(x: int) -> int:
    return x * 2


def add(x: int, y: int) -> int:
    return x + y


def test_save_with_fallback():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    nd = pipe.add_component("double", double, x=a)
    nn = pipe.add_component("negate", negative, x=a)
    fb = pipe.use_first_of("fill-operand", b, nn)
    pipe.add_component("add", add, x=nd, y=fb)

    cfg = pipe.get_config()
    json = cfg.model_dump_json(exclude_none=True)
    print(json)
    c2 = PipelineConfig.model_validate_json(json)

    p2 = Pipeline.from_config(c2)

    # 3 * 2 + -3 = 3
    assert p2.run("fill-operand", "add", a=3) == (-3, 3)


def test_hash_validate():
    pipe = Pipeline()
    msg = pipe.create_input("msg", str)

    pfx = Prefixer("scroll named ")
    pipe.add_component("prefix", pfx, msg=msg)

    cfg = pipe.get_config()
    print("initial config:", cfg.model_dump_json(indent=2))
    assert cfg.meta.hash is not None
    cfg.components["prefix"].config["prefix"] = "scroll called "  # type: ignore
    print("modified config:", cfg.model_dump_json(indent=2))

    with warns(PipelineWarning):
        Pipeline.from_config(cfg)


def test_alias_input():
    "alias an input node"
    pipe = Pipeline()
    user = pipe.create_input("user", int, str)

    pipe.alias("person", user)

    cfg = pipe.get_config()

    p2 = Pipeline.from_config(cfg)
    assert p2.run("person", user=32) == 32


def test_alias_node():
    pipe = Pipeline()
    a = pipe.create_input("a", int)
    b = pipe.create_input("b", int)

    nd = pipe.add_component("double", double, x=a)
    na = pipe.add_component("add", add, x=nd, y=b)
    pipe.alias("result", na)

    assert pipe.run("result", a=5, b=7) == 17

    cfg = pipe.get_config()
    p2 = Pipeline.from_config(cfg)
    assert p2.run("result", a=5, b=7) == 17
