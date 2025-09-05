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

from lenskit.als import BiasedMFScorer
from lenskit.cli import lenskit
from lenskit.data import Dataset
from lenskit.pipeline.nodes import ComponentInstanceNode


def test_train_config(tmpdir: Path, ml_ds: Dataset):
    ds_path = tmpdir / "ml-data"
    ml_ds.save(ds_path)

    out_file = tmpdir / "als.pkl.gz"

    runner = CliRunner()
    result = runner.invoke(
        lenskit,
        [
            "train",
            "--config",
            "pipelines/als-explicit.toml",
            "-o",
            fspath(out_file),
            fspath(ds_path),
        ],
    )

    assert result.exit_code == 0
    assert out_file.exists()

    with xopen(out_file, "rb") as pf:
        pipe = pickle.load(pf)

    node = pipe.node("scorer")
    assert isinstance(node, ComponentInstanceNode)
    assert isinstance(node.component, BiasedMFScorer)
