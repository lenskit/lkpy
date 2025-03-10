# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from pydantic.dataclasses import dataclass as pydantic_dataclass

from pytest import mark, raises

from lenskit.diagnostics import PipelineError
from lenskit.pipeline import Component, Pipeline, PipelineBuilder
from lenskit.pipeline.nodes import ComponentInstanceNode


@dataclass
class PrefixConfigDC:
    prefix: str = "UNDEFINED"


class PrefixConfigM(BaseModel):
    prefix: str = "UNDEFINED"


@pydantic_dataclass
class PrefixConfigPYDC:
    prefix: str = "UNDEFINED"


class PrefixerDC(Component[str]):
    config: PrefixConfigDC

    def __call__(self, msg: str) -> str:
        return self.config.prefix + msg


class PrefixerM(Component[str]):
    config: PrefixConfigM

    def __call__(self, msg: str) -> str:
        return self.config.prefix + msg


class PrefixerPYDC(Component[str]):
    config: PrefixConfigPYDC

    def __call__(self, msg: str) -> str:
        return self.config.prefix + msg


@dataclass
class SuffixConfig:
    suffix: str


class Suffixer(Component[str]):
    config: SuffixConfig

    def __call__(self, msg: str) -> str:
        return msg + self.config.suffix


def lowercase(msg: str) -> str:
    "Lowercase a string."
    return msg.lower()


@mark.parametrize("prefixer", [PrefixerDC, PrefixerM, PrefixerPYDC])
def test_config_reuse_only(prefixer: type[Component]):
    ccls = prefixer.config_class()  # type: ignore
    assert ccls is not None

    build = PipelineBuilder("test")
    msg = build.create_input("msg", str)
    build.add_component("prefix", prefixer, ccls(prefix="hello "), msg=msg)

    p1 = build.build()
    p2 = build.build()

    n1 = p1.node("prefix")
    assert isinstance(n1, ComponentInstanceNode)
    c1 = n1.component

    n2 = p2.node("prefix")
    assert isinstance(n2, ComponentInstanceNode)
    c2 = n2.component

    assert c1 is c2

    assert p1.run("prefix", msg="woozle") == "hello woozle"


@mark.parametrize("prefixer", [PrefixerDC, PrefixerM, PrefixerPYDC])
def test_config_reuse_one(prefixer: type[Component]):
    ccls = prefixer.config_class()  # type: ignore
    assert ccls is not None

    build = PipelineBuilder("test")
    msg = build.create_input("msg", str)
    pm = build.add_component("prefix", prefixer, ccls(prefix="hello "), msg=msg)
    build.add_component("suffix", Suffixer, {"suffix": " for now"}, msg=pm)

    p1 = build.build()

    build.replace_component("prefix", prefixer, ccls(prefix="goodbye "))
    p2 = build.build()

    n1 = p1.node("prefix")
    assert isinstance(n1, ComponentInstanceNode)
    c1 = n1.component

    n2 = p2.node("prefix")
    assert isinstance(n2, ComponentInstanceNode)
    c2 = n2.component

    assert c1 is not c2

    n1 = p1.node("suffix")
    assert isinstance(n1, ComponentInstanceNode)
    c1 = n1.component

    n2 = p2.node("suffix")
    assert isinstance(n2, ComponentInstanceNode)
    c2 = n2.component

    assert c1 is c2

    assert p1.run("suffix", msg="woozle") == "hello woozle for now"
    assert p2.run("suffix", msg="woozle") == "goodbye woozle for now"
