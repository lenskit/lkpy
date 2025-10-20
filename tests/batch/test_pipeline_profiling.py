# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pandas as pd

from pytest import mark

from lenskit.basic import BiasScorer
from lenskit.batch import BatchPipelineRunner
from lenskit.pipeline import PipelineProfiler, topn_pipeline
from lenskit.splitting.split import TTSplit

from .test_batch_pipeline import MLB, ml_split, mlb


@mark.parametrize(("ncpus"), [1, 2])
@mark.eval
def test_bias_batch(tmpdir: Path, ml_split: TTSplit, ncpus: int | None):
    algo = BiasScorer(damping=5)
    pipeline = topn_pipeline(algo, predicts_ratings=True, n=20)
    pipeline.train(ml_split.train)

    file = tmpdir / "profile.csv"

    with PipelineProfiler(pipeline, file) as profiler:
        runner = BatchPipelineRunner(n_jobs=ncpus, profiler=profiler)
        runner.recommend()
        runner.predict()

        _results = runner.run(pipeline, ml_split.test)

    prof = pd.read_csv(file)
    assert list(prof.columns) == pipeline.component_names()
    assert len(prof) == len(ml_split.test)
    # is the scorer the right time?
    assert prof["scorer"].mean() > 1.0e-5
    assert prof["scorer"].mean() < 0.2
