# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from lenskit.als import BiasedMFScorer, ImplicitMFScorer
from lenskit.basic import BiasScorer, FallbackScorer, TopNRanker
from lenskit.config import load_config_data
from lenskit.pipeline import Pipeline, PipelineConfig
from lenskit.pipeline.nodes import ComponentInstanceNode

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
    assert isinstance(node, ComponentInstanceNode)
    assert isinstance(node.component, ImplicitMFScorer)

    node = als_pipe.node("recommender")
    assert isinstance(node, ComponentInstanceNode)
    assert isinstance(node.component, TopNRanker)


def test_apply_predictor_config():
    als_file = config_dir / "als-explicit.toml"
    als_pipe = Pipeline.load_config(als_file)

    node = als_pipe.node("scorer")
    assert isinstance(node, ComponentInstanceNode)
    assert isinstance(node.component, BiasedMFScorer)

    node = als_pipe.node("recommender")
    assert isinstance(node, ComponentInstanceNode)
    assert isinstance(node.component, TopNRanker)

    node = als_pipe.node("rating-predictor")
    assert isinstance(node, ComponentInstanceNode)
    assert isinstance(node.component, FallbackScorer)

    node = als_pipe.node("fallback-predictor")
    assert isinstance(node, ComponentInstanceNode)
    assert isinstance(node.component, BiasScorer)


def test_apply_no_fallback():
    config = {
        "meta": {"name": "bias"},
        "options": {"base": "std:topn-predict", "fallback_predictor": False},
        "components": {"scorer": {"class": "lenskit.basic.BiasScorer"}},
    }
    bias_pipe = Pipeline.from_config(config)

    node = bias_pipe.node("scorer")
    assert isinstance(node, ComponentInstanceNode)
    assert isinstance(node.component, BiasScorer)

    node = bias_pipe.node("recommender")
    assert isinstance(node, ComponentInstanceNode)
    assert isinstance(node.component, TopNRanker)

    node = bias_pipe.node("rating-predictor")
    assert isinstance(node, ComponentInstanceNode)
    assert isinstance(node.component, BiasScorer)

    node = bias_pipe.node("fallback-predictor", missing=None)
    assert node is None
