# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import json

from lenskit.pipeline import Pipeline
from lenskit.pipeline.components import Component


class Prefixer(Component):
    prefix: str

    def __init__(self, prefix: str = "hello"):
        self.prefix = prefix

    def __call__(self, msg: str) -> str:
        return self.prefix + msg


def test_auto_config_roundtrip():
    comp = Prefixer("FOOBIE BLETCH")

    cfg = comp.get_config()
    assert "prefix" in cfg

    c2 = Prefixer.from_config(cfg)
    assert c2 is not comp
    assert c2.prefix == comp.prefix


def test_pipeline_config():
    comp = Prefixer("scroll named ")

    pipe = Pipeline()
    msg = pipe.create_input("msg", str)
    pipe.add_component("prefix", comp, msg=msg)

    assert pipe.run(msg="FOOBIE BLETCH") == "scroll named FOOBIE BLETCH"

    config = pipe.component_configs()
    print(json.dumps(config, indent=2))

    assert "prefix" in config
    assert config["prefix"]["prefix"] == "scroll named "
