# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

import click

from lenskit.data.amazon import load_amazon_ratings
from lenskit.data.movielens import load_movielens
from lenskit.logging import get_logger

_log = get_logger(__name__)


@click.command("convert")
@click.option("--movielens", "format", flag_value="movielens", help="convert MovieLens data")
@click.option("--amazon", "format", flag_value="amazon", help="convert Amazon rating data")
@click.argument("src", type=Path)
@click.argument("dst", type=Path)
def convert(format: str | None, src: Path, dst: Path):
    """
    Convert data into the LensKit native format.
    """

    log = _log.bind(src=str(src))

    match format:
        case None:
            _log.error("no data format specified")
            raise click.UsageError("no data format specified")
        case "movielens":
            log.info("loading MovieLens data")
            data = load_movielens(src)
        case "amazon":
            data = load_amazon_ratings(src)
        case _:
            raise ValueError(f"unknown data format {format}")

    log = _log.bind(dst=str(dst))
    log.info("saving data in native format")
    data.save(dst)
