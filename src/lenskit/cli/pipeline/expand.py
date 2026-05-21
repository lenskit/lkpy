# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

import click

from lenskit.logging import get_logger, stdout_console

from ._group import pipeline
from ._load import PipelineLoadSpec, wants_pipeline_config

_log = get_logger(__name__)


@pipeline.command
@click.option(
    "-o",
    "--output",
    "out_file",
    type=Path,
    metavar="FILE",
    help="Save expanded pipeline configuration to FILE.",
)
@wants_pipeline_config
def expand(
    pipe_cfg: PipelineLoadSpec,
    out_file: Path | None,
):
    """
    Expand a pipeline configuration into fully-formed pipeline and save or print
    the resulting configuration.
    """
    console = stdout_console()

    pipe = pipe_cfg.load_pipeline()

    if out_file is not None:
        with out_file.open("wt") as jf:
            jf.write(pipe.config.model_dump_json())
            jf.write("\n")
    else:
        console.print_json(pipe.config.model_dump_json())
