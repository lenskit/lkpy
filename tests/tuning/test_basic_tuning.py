# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest import fixture, mark, skip

from lenskit.data import from_interactions_df
from lenskit.splitting import sample_records
from lenskit.tuning import TuningSpec


@mark.parametrize("backend", ["optuna", "ray"])
@fixture
def tuner_class(backend):
    match backend:
        case "optuna":
            from lenskit.tuning import PipelineTuner

            return PipelineTuner
        case "ray":
            try:
                from lenskit.tuning import RayPipelineTuner
            except ImportError:
                skip("ray not available")

            return RayPipelineTuner


@mark.slow
@mark.realdata
def test_tune_bias(ml_100k, tmpdir, tuner_class):
    spec = TuningSpec.load(Path("pipelines/bias-search.toml"))
    tuner = tuner_class(spec, Path(tmpdir))
    split = sample_records(from_interactions_df(ml_100k), 20000)

    tuner.set_data(split.train, split.test)

    result = tuner.run()
    print(result.best_result())
