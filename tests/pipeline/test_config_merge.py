# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.basic import BiasScorer, PopScorer
from lenskit.basic.candidates import TrainingItemsCandidateSelector
from lenskit.pipeline import Pipeline, topn_pipeline
from lenskit.pipeline.config import PipelineConfigFragment


def test_merge_simple_setting():
    pipe = topn_pipeline(PopScorer, n=100)
    cfg = pipe.config
    assert cfg.components["scorer"].config is not None
    assert cfg.components["scorer"].config["score"] == "quantile"

    c2 = pipe.config.merge_component_configs({"scorer": {"score": "rank"}})
    assert c2 is not cfg
    assert c2.components["scorer"].config is not None
    assert c2.components["scorer"].config["score"] == "rank"


def test_merge_deep_setting():
    pipe = topn_pipeline(BiasScorer, n=100)
    cfg = pipe.config
    assert cfg.components["scorer"].config is not None
    assert cfg.components["scorer"].config["damping"] == 0.0

    c2 = pipe.config.merge_component_configs({"scorer": {"damping": {"user": 150, "item": 4}}})
    assert c2 is not cfg
    assert c2.components["scorer"].config is not None
    assert isinstance(c2.components["scorer"].config["damping"], dict)
    assert c2.components["scorer"].config["damping"]["user"] == 150
    assert c2.components["scorer"].config["damping"]["item"] == 4


def test_merge_override():
    pipe = topn_pipeline(BiasScorer, n=100)
    cfg = pipe.config

    update = PipelineConfigFragment.model_validate({"options": {"base": "std:topn"}})
    c2 = cfg.merge_config(update)

    assert c2.options is None or c2.options.base is None


def test_configure_component():
    pipe = topn_pipeline(BiasScorer, n=100)
    cfg = pipe.config

    update = PipelineConfigFragment.model_validate(
        {"components": {"candidate-selector": {"config": {"exclude": None}}}}
    )
    c2 = cfg.merge_config(update)

    comp = c2.components["candidate-selector"]
    assert comp.code == "lenskit.basic.candidates:TrainingItemsCandidateSelector"
    assert comp.config is not None
    assert comp.config["exclude"] is None

    pipe = Pipeline.from_config(c2)
    comp = pipe.component("candidate-selector")
    assert isinstance(comp, TrainingItemsCandidateSelector)
    assert comp.config.exclude is None
