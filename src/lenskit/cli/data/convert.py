# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

import click

from lenskit.logging import get_logger

_log = get_logger(__name__)


@click.command("convert")
@click.option("--movielens", "format", flag_value="movielens", help="Convert MovieLens data.")
@click.option("--amazon", "format", flag_value="amazon", help="Convert Amazon rating data.")
@click.option("--ms-web", "format", flag_value="ms-web", help="Convert MSWeb visit logs.")
@click.option("--steam", "format", flag_value="steam", help="Convert Steam interaction data")
@click.option(
    "--item-lists", is_flag=True, help="Convert to an ItemListCollection instead of Dataset."
)
@click.option(
    "--summary/--no-summary",
    "summary",
    default=True,
    help="Include a summary Markdown file in the exported dataset.",
)
@click.argument("src", type=Path, nargs=-1, required=True)
@click.argument("dst", type=Path, required=True)
def convert(
    format: str | None, summary: bool, src: list[Path], dst: Path, item_lists: bool = False
):
    """
    Convert data into the LensKit native format.

    By default, this will create a LensKit Dataset.  With --item-lists, it will
    extract the default interactions into an ItemListCollection and save it in
    native format.
    """

    log = _log.bind(src=[str(p) for p in src])

    match format:
        case None:
            _log.error("no data format specified")
            raise click.UsageError("no data format specified")
        case "movielens":
            from lenskit.data.sources.movielens import load_movielens

            log.info("loading MovieLens data")
            if len(src) != 1:
                log.error("received %d source paths, MovieLens only takes one", len(src))

            data = load_movielens(src[0])
        case "amazon":
            from lenskit.data.sources.amazon import load_amazon_ratings

            data = load_amazon_ratings(*src)
        case "ms-web":
            from lenskit.data.sources.msweb import load_ms_web

            data = load_ms_web(src[0])
        case "steam":
            from lenskit.data.sources.steam import load_steam

            data = load_steam(src[0])
        case _:
            raise ValueError(f"unknown data format {format}")

    log = _log.bind(dst=str(dst), name=data.name)
    if item_lists:
        icls = data.default_interaction_class()
        log.info("extracting %s data", icls)
        rels = data.interactions(icls).matrix()
        ilc = rels.item_lists()
        log.info("saving to native Parquet format")
        ilc.save_parquet(dst)
    else:
        log.info("saving data in native format")
        data.save(dst, summary=summary)
