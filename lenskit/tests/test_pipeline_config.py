from lenskit.pipeline.components import AutoConfig


class MyComponent(AutoConfig):
    param: str

    def __init__(self, param: str = "hello"):
        self.param = param


def test_auto_config_roundtrip():
    comp = MyComponent("FOOBIE BLETCH")

    cfg = comp.get_config()
    assert "param" in cfg

    c2 = MyComponent.from_config(cfg)
    assert c2 is not comp
    assert c2.param == comp.param
