# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic.dataclasses import dataclass as pydantic_dataclass

from pytest import mark

from lenskit.pipeline import PipelineBuilder
from lenskit.pipeline.components import Component, ComponentConstructor
from lenskit.pipeline.nodes import ComponentConstructorNode


class EarlyConfig(Component[str]):
    config: PrefixConfigDC

    def __call__(self, msg: str) -> str:
        return self.config.prefix + msg


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


# make sure it works with sub-sub-classes
class PrefixerM2(PrefixerM):
    config: PrefixConfigM


class PrefixerPYDC(Component[str]):
    config: PrefixConfigPYDC

    def __call__(self, msg: str) -> str:
        return self.config.prefix + msg


@mark.parametrize("prefixer", [PrefixerDC, PrefixerM, PrefixerPYDC, PrefixerM2])
def test_config_setup(prefixer: type[Component]):
    ccls = prefixer.config_class()  # type: ignore
    assert ccls is not None

    comp = prefixer()
    assert isinstance(comp.config, ccls)


@mark.parametrize("prefixer", [PrefixerDC, PrefixerM, PrefixerPYDC])
def test_auto_config_roundtrip(prefixer: type[Component]):
    comp = prefixer(prefix="FOOBIE BLETCH")

    cfg = comp.config
    cfg_data = comp.dump_config()
    assert "prefix" in cfg_data

    c2 = prefixer(cfg)
    assert c2 is not comp
    assert c2.config.prefix == comp.config.prefix

    c3 = prefixer(prefixer.validate_config(cfg_data))
    assert c3 is not comp
    assert c3.config.prefix == comp.config.prefix


@mark.parametrize("prefixer", [PrefixerDC, PrefixerM, PrefixerPYDC])
def test_pipeline_config(prefixer: ComponentConstructor[Any, str]):
    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)
    pn = pipe.add_component("prefix", prefixer, {"prefix": "scroll named "}, msg=msg)
    assert isinstance(pn, ComponentConstructorNode)
    assert pn.constructor == prefixer
    assert getattr(pn.config, "prefix") == "scroll named "

    pipe = pipe.build()
    assert pipe.run("prefix", msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH"

    config = pipe.config.components
    print(TypeAdapter(dict).dump_json(config, indent=2))

    assert "prefix" in config
    assert config["prefix"].config
    assert config["prefix"].config["prefix"] == "scroll named "


@mark.parametrize("prefixer", [PrefixerDC, PrefixerM, PrefixerPYDC])
def test_pipeline_config_roundtrip(prefixer: type[Component]):
    comp = prefixer(prefix="scroll named ")

    pipe = PipelineBuilder()
    msg = pipe.create_input("msg", str)
    pipe.add_component("prefix", comp, msg=msg)
    pipe.default_component("prefix")

    assert pipe.build().run("prefix", msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH"

    config = pipe.build_config()
    print(config.model_dump_json(indent=2))

    p2 = PipelineBuilder.from_config(config)
    assert p2.node("prefix", missing="none") is not None
    assert p2.build().run(msg="READ ME") == "scroll named READ ME"
