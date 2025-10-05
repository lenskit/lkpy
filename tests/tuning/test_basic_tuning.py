# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest import mark

from lenskit.data import from_interactions_df
from lenskit.splitting import sample_records
from lenskit.tuning import PipelineTuner, TuningSpec


@mark.slow
def test_tune_bias(ml_100k, tmpdir):
    spec = TuningSpec.load(Path("pipelines/bias-search.toml"))
    tuner = PipelineTuner(spec, Path(tmpdir))
    split = sample_records(from_interactions_df(ml_100k), 20000)

    tuner.set_data(split.train, split.test)

    tuner.run()
    print(tuner.best_result())
