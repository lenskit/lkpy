# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pandas as pd

from lenskit.pipeline import PipelineBuilder, PipelineProfiler

from .test_pipeline_clone import PrefixConfig, Prefixer


def test_basic_profile(tmpdir: Path):
    builder = PipelineBuilder()
    msg = builder.create_input("msg", str)
    builder.add_component("prefix", Prefixer, PrefixConfig(prefix="scroll named "), msg=msg)
    builder.default_component("prefix")
    pipe = builder.build()

    pipe_file = tmpdir / "profile.csv"
    with PipelineProfiler(pipe, pipe_file) as profiler:
        pipe.run("prefix", msg="hello", _profile=profiler)

    assert pipe_file.exists()
    print(pipe_file.read_text("utf8"))
    df = pd.read_csv(pipe_file)
    assert df.columns == ["prefix"]
    assert len(df) == 1

    # very short time - less than 100ms
    assert df.loc[0, "prefix"] < 100000
