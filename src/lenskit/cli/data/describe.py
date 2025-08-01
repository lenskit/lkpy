# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import sys
from io import StringIO
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown

from lenskit.data import Dataset, load_amazon_ratings, load_movielens
from lenskit.data._summary import save_stats
from lenskit.data.sources.steam import load_steam
from lenskit.logging import get_logger

_log = get_logger(__name__)


@click.command("describe")
@click.option("--movielens", "format", flag_value="movielens", help="Describe MovieLens data.")
@click.option("--amazon", "format", flag_value="amazon", help="Describe Amazon rating data.")
@click.option("--steam", "format", flag_value="steam", help="Describe Steam interaction data.")
@click.option("--markdown", is_flag=True, help="output raw Markdown")
@click.argument("path", type=Path, nargs=-1, required=True)
def describe(format: str | None, markdown: bool, path: list[Path]):
    """
    Describe a data set.
    """

    if len(path) == 1:
        log = _log.bind(path=str(path[0]))
    else:
        log = _log.bind(path=[str(p) for p in path])

    match format:
        case None:
            log.info("loading LensKit native data")
            data = Dataset.load(path[0])
        case "movielens":
            log.info("loading MovieLens data")
            data = load_movielens(path[0])
        case "amazon":
            log.info("loading Amazon data")
            data = load_amazon_ratings(path[0])
        case "steam":
            log.info("loading Steam data")
            data = load_steam(*path)
        case _:
            raise ValueError(f"unknown data format {format}")

    console = Console()

    if markdown:
        save_stats(data, sys.stdout)
    else:
        out = StringIO()
        save_stats(data, out)
        console.print(Markdown(out.getvalue()), width=min(console.width, 80))
