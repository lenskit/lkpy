# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from lenskit.als import ImplicitMFScorer
from lenskit.basic import TopNRanker
from lenskit.config import load_config_data
from lenskit.pipeline import Pipeline, PipelineConfig
from lenskit.pipeline.nodes import ComponentConstructorNode

config_dir = Path("pipelines")


def test_load_config():
    als_file = config_dir / "als-implicit.toml"
    als_cfg = load_config_data(als_file, PipelineConfig)
    assert isinstance(als_cfg, PipelineConfig)
    assert als_cfg.meta.name == "als-implicit"


def test_apply_scorer_config():
    als_file = config_dir / "als-implicit.toml"
    als_pipe = Pipeline.load_config(als_file)

    node = als_pipe.node("scorer")
    assert isinstance(node, ComponentConstructorNode)
    assert node.constructor == ImplicitMFScorer

    node = als_pipe.node("recommender")
    assert isinstance(node, ComponentConstructorNode)
    assert node.constructor == TopNRanker
