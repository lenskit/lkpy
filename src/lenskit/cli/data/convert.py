# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path

import click

from lenskit.data.amazon import load_amazon_ratings
from lenskit.data.collection._list import ListILC
from lenskit.data.movielens import load_movielens
from lenskit.logging import get_logger

_log = get_logger(__name__)


@click.command("convert")
@click.option("--movielens", "format", flag_value="movielens", help="Convert MovieLens data.")
@click.option("--amazon", "format", flag_value="amazon", help="Convert Amazon rating data.")
@click.option(
    "--item-lists", is_flag=True, help="Convert to an ItemListCollection instead of Dataset."
)
@click.argument("src", nargs=-1, required=True, type=Path)
@click.argument("dst", type=Path)
def convert(format: str | None, src: list[Path], dst: Path, item_lists: bool = False):
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
            if len(src) != 1:
                log.error("received %d source paths, MovieLens only takes one", len(src))
            data = load_movielens(src[0])
        case "amazon":
            data = load_amazon_ratings(*src)
        case _:
            raise ValueError(f"unknown data format {format}")

    log = _log.bind(dst=str(dst), name=data.name)
    if item_lists:
        icls = data.default_interaction_class()
        log.info("extracting %s data", icls)
        rels = data.interactions(icls).matrix()
        ilc: ListILC = ListILC.from_dict(
            {(rels.row_vocabulary.id(i),): rels.row_items(number=i) for i in range(rels.n_rows)},  # type: ignore
            f"{rels.row_type}_id",
        )
        log.info("saving to native Parquet format")
        ilc.save_parquet(dst)
    else:
        log.info("saving data in native format")
        data.save(dst)
