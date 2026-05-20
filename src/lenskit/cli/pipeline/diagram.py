# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Literal

import click
from rich.syntax import Syntax

from lenskit.logging import get_logger, stdout_console
from lenskit.pipeline import MermaidDiagrammer

from ._group import pipeline
from ._load import PipelineLoadSpec, wants_pipeline_config

_log = get_logger(__name__)


@pipeline.command
@click.option(
    "--mermaid", "format", flag_value="mermaid", default="mermaid", help="Save in Mermaid syntax."
)
@click.option(
    "--graphviz",
    "format",
    flag_value="graphviz",
    help="Save in Graphviz (DOT) syntax.",
)
@click.option("--render", "render_tty", is_flag=True, help="Render to terminal (requires termaid).")
@click.option("--label-edges/--no-label-edges", help="Label edges between nodes.")
@click.option(
    "-o",
    "--output",
    "out_file",
    type=Path,
    metavar="FILE",
    help="Save pipeline diagram to FILE.",
)
@wants_pipeline_config
def diagram(
    pipe_cfg: PipelineLoadSpec,
    out_file: Path | None,
    label_edges: bool,
    render_tty: bool,
    format: Literal["graphviz", "mermaid"],
):
    """
    Render a diagram of the pipeline.
    """
    console = stdout_console()

    if format != "mermaid":
        _log.error("graph format %s not yet supported", format)
        raise SystemExit(2)

    pipe = pipe_cfg.load_pipeline()

    diag = MermaidDiagrammer()
    diag.label_edges = label_edges

    diag.render_pipeline(pipe)
    mmd = diag.text()

    if out_file is not None:
        out_file.write_text(mmd)

    if render_tty:
        import termaid

        d2 = MermaidDiagrammer("TB", for_tty=True)
        d2.render_pipeline(pipe)

        console.print(termaid.render_rich(d2.text()))

    elif out_file is None:
        if console.is_terminal:
            console.print(Syntax(mmd, "mermaid"))
        elif out_file is None:
            print(mmd)
