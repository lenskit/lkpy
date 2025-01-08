# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic.dataclasses import dataclass as pydantic_dataclass

from pytest import mark

from lenskit.pipeline import Pipeline
from lenskit.pipeline.components import Component


@dataclass
class PrefixConfigDC:
    prefix: str = "UNDEFINED"


class PrefixConfigM(BaseModel):
    prefix: str = "UNDEFINED"


@pydantic_dataclass
class PrefixConfigPYDC:
    prefix: str = "UNDEFINED"


class PrefixerDC(Component):
    config: PrefixConfigDC

    def __call__(self, msg: str) -> str:
        return self.config.prefix + msg


class PrefixerM(Component):
    config: PrefixConfigM

    def __call__(self, msg: str) -> str:
        return self.config.prefix + msg


class PrefixerPYDC(Component):
    config: PrefixConfigPYDC

    def __call__(self, msg: str) -> str:
        return self.config.prefix + msg


@mark.parametrize("prefixer", [PrefixerDC, PrefixerM, PrefixerPYDC])
def test_config_setup(prefixer: type[Component]):
    ccls = prefixer._config_class()  # type: ignore
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
def test_pipeline_config(prefixer: type[Component]):
    comp = prefixer(prefix="scroll named ")

    pipe = Pipeline()
    msg = pipe.create_input("msg", str)
    pipe.add_component("prefix", comp, msg=msg)

    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH"

    config = pipe.component_configs()
    print(json.dumps(config, indent=2))

    assert "prefix" in config
    assert config["prefix"]["prefix"] == "scroll named "


@mark.parametrize("prefixer", [PrefixerDC, PrefixerM, PrefixerPYDC])
def test_pipeline_config_roundtrip(prefixer: type[Component]):
    comp = prefixer(prefix="scroll named ")

    pipe = Pipeline()
    msg = pipe.create_input("msg", str)
    pipe.add_component("prefix", comp, msg=msg)

    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH"

    config = pipe.get_config()
    print(config.model_dump_json(indent=2))

    p2 = Pipeline.from_config(config)
    assert p2.node("prefix", missing="none") is not None
    assert p2.run(msg="READ ME") == "scroll named READ ME"
