# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import json
import pickle
from os import fspath
from pathlib import Path

from click.testing import CliRunner
from xopen import xopen

from pytest import mark

from lenskit.als import BiasedMFScorer
from lenskit.cli import lenskit
from lenskit.data import Dataset
from lenskit.data.adapt import from_interactions_df
from lenskit.pipeline.config import PipelineConfig
from lenskit.pipeline.nodes import ComponentInstanceNode


@mark.slow
@mark.realdata
def test_tune_cli(tmpdir: Path, ml_100k):
    ml_ds = from_interactions_df(ml_100k)
    ds_path = tmpdir / "ml-data"
    ml_ds.save(ds_path)

    train_file = tmpdir / "ml-data.train"
    test_file = tmpdir / "ml-data.test.parquet"
    out_dir = tmpdir / "bias-tune"

    runner = CliRunner()
    result = runner.invoke(
        lenskit, ["data", "split", "--fraction=0.2", "--min-train-interactions=5", fspath(ds_path)]
    )
    assert result.exit_code == 0

    result = runner.invoke(
        lenskit,
        [
            "tune",
            "-T",
            fspath(train_file),
            "-V",
            fspath(test_file),
            "--save-pipeline",
            fspath(tmpdir / "pipeline.json"),
            "pipelines/bias-search.toml",
            fspath(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert out_dir.exists()
    assert (out_dir / "result.json").exists()
    assert (tmpdir / "pipeline.json").exists()

    cfg = PipelineConfig.model_validate_json((tmpdir / "pipeline.json").read_text())
    res = json.loads((out_dir / "result.json").read_text())

    s_cfg = cfg.components["scorer"].config
    assert s_cfg is not None
    assert s_cfg["damping"] == res["config"]["damping"]
