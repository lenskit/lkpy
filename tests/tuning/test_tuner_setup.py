# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from lenskit.tuning import PipelineTuner
from lenskit.tuning.spec import TuningSpec


def test_tuner_spec():
    spec = TuningSpec.load(Path("pipelines/iknn-explicit-search.toml"))
    assert len(spec.space) == 1
    assert spec.component_name == "scorer"


def test_tuner_space():
    spec = TuningSpec.load(Path("pipelines/iknn-explicit-search.toml"))
    tuner = PipelineTuner(spec)

    space = tuner.search_space()
    assert space is not None
