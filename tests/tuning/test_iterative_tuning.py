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


@fixture(params=["optuna", "ray"])
def tuner_class(request):
    match request.param:
        case "optuna":
            try:
                import optuna
            except ImportError:
                skip("optuna not available")

            from lenskit.tuning import PipelineTuner

            yield PipelineTuner
        case "ray":
            try:
                from lenskit.tuning import RayPipelineTuner
            except ImportError:
                skip("ray not available")

            yield RayPipelineTuner


@mark.slow
@mark.realdata
@mark.parametrize("version", ["implicit", "explicit"])
def test_tune_als(ml_100k, tmpdir, tuner_class, version: str):
    "test optimizing an iterative model"
    spec = TuningSpec.load(Path(f"pipelines/als-{version}-search.toml"))
    spec.search.method = "random"
    spec.search.max_points = 10
    spec.search.max_epochs = 5
    spec.search.plateau_min_rel_improvement = 0.05
    spec.search.median_min_trials = 3

    tpath = Path(tmpdir)

    tuner = tuner_class(spec, tpath)
    assert tuner.iterative
    split = sample_records(from_interactions_df(ml_100k), 20000)

    tuner.set_data(split.train, split.test)

    result = tuner.run()
    print(result.best_result())
    print(result.best_config())
    assert result.iterative
    assert result.best_config()["epochs"] <= 5

    if tuner_class.__name__ == "PipelineTuner":
        assert (tpath / "trials.csv").exists()
        assert (tpath / "trial-epochs.csv").exists()
