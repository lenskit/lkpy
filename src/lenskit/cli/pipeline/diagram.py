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
from lenskit.pipeline import Component, MermaidDiagrammer, Pipeline, topn_pipeline
from lenskit.pipeline._types import import_path_string

from ._group import pipeline

_log = get_logger(__name__)


@pipeline.command
@click.option(
    "-C", "--scorer-class", metavar="CLS", help="Create a default pipeline with scorer CLS"
)
@click.option(
    "-c", "--config", type=Path, metavar="FILE", help="Load pipeline configuration from FILE."
)
@click.option(
    "--rating-predictor",
    is_flag=True,
    help="Include rating prediction in the pipeline capabilities.",
)
@click.option("-n", "--list-length", type=int, help="Default list length for pipeline ranker.")
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
@click.option("--plain-nodes", is_flag=True, help="Use plain node text (easier for terminals).")
@click.option(
    "-o",
    "--output",
    "out_file",
    type=Path,
    metavar="FILE",
    help="Save pipeline diagram to FILE.",
)
def diagram(
    scorer_class: str | None,
    config: Path | None,
    out_file: Path | None,
    list_length: int | None,
    label_edges: bool,
    plain_nodes: bool,
    render_tty: bool,
    format: Literal["graphviz", "mermaid"],
    rating_predictor: bool,
):
    """
    Render a diagram of the pipeline.
    """
    console = stdout_console()
    if config is not None:
        if scorer_class is not None:
            _log.error("cannot specify both scorer class and configuration file")
            raise SystemExit(3)

        _log.info("loading pipeline from %s", config)
        pipe = Pipeline.load_config(config)

    elif scorer_class is not None:
        _log.info("creating pipeline for %s", config)
        scorer = import_path_string(scorer_class)
        assert issubclass(scorer, Component)
        pipe = topn_pipeline(scorer, predicts_ratings=rating_predictor, n=list_length)

    else:
        _log.error("no scorer specified")
        raise SystemExit(5)

    if format != "mermaid":
        _log.error("graph format %s not yet supported", format)
        raise SystemExit(2)

    diag = MermaidDiagrammer()
    diag.label_edges = label_edges and not render_tty
    diag.plain_nodes = plain_nodes or render_tty

    diag.render_pipeline(pipe)
    mmd = diag.text()

    if out_file is not None:
        out_file.write_text(mmd)

    if render_tty:
        import termaid

        console.print(termaid.render_rich(mmd))

    elif out_file is None:
        if console.is_terminal:
            console.print(Syntax(mmd, "mermaid"))
        elif out_file is None:
            print(mmd)
