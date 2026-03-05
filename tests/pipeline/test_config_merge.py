# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.basic import BiasScorer, PopScorer
from lenskit.pipeline.common import topn_pipeline


def test_merge_simple_setting():
    pipe = topn_pipeline(PopScorer, n=100)
    cfg = pipe.config
    assert cfg.components["scorer"].config["score"] == "quantile"

    c2 = pipe.config.merge_component_configs({"scorer": {"score": "rank"}})
    assert c2 is not cfg
    assert c2.components["scorer"].config["score"] == "rank"


def test_merge_deep_setting():
    pipe = topn_pipeline(BiasScorer, n=100)
    cfg = pipe.config
    assert cfg.components["scorer"].config["damping"] == 0.0

    c2 = pipe.config.merge_component_configs({"scorer": {"damping": {"user": 150, "item": 4}}})
    assert c2 is not cfg
    assert c2.components["scorer"].config["damping"]["user"] == 150
    assert c2.components["scorer"].config["damping"]["item"] == 4
