# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Literal

import click

from lenskit.data import ItemListCollection
from lenskit.data.dataset import Dataset
from lenskit.logging import get_logger
from lenskit.random import random_generator

_log = get_logger(__name__)


@click.command("subset")
@click.option(
    "--item-lists", is_flag=True, help="Write an ItemListCollection instead of a Dataset."
)
@click.option(
    "--layout",
    type=click.Choice(["native", "flat"]),
    default="native",
    help="Specify layout for item list output.",
)
@click.option(
    "--sample-rows",
    type=int,
    metavar="N",
    help="Subset to at most N rows of the interaction data",
)
@click.argument("src", required=True, type=Path)
@click.argument("dst", required=True, type=Path)
def subset(
    src: Path,
    dst: Path,
    sample_rows: int | None,
    item_lists: bool,
    layout: Literal["native", "flat"],
):
    """
    Subset a LensKit dataset.

    The input SRC can be either a LensKit native dataset or a Parquet file
    containing an item list colleciton in native format.  The "rows" in this
    subsetting operation are entire rows of the interaction matrix, typically
    users.
    """
    log = _log.bind(src=str(src))

    dataset: Dataset | None = None
    lists: ItemListCollection
    if src.is_file():
        log.info("loading ItemListCollection")
        lists = ItemListCollection.load_parquet(src)
    else:
        log.debug("loading Dataset")
        dataset = Dataset.load(src)
        log.info("extracting default interactions")
        if len(dataset.schema.relationships) > 1:
            log.warn("Dataset has multiple relationships, only default interactions are subset")

        # TODO: support non-default interactions / multiple interaction sets
        lists = dataset.interactions().matrix().to_ilc()

    if sample_rows is not None:
        lists = _subset_lists(lists, sample_rows)

    log = _log.bind(dst=str(dst))
    if item_lists:
        log.info("saving ItemListCollection Parquet in %s layout", layout)
        lists.save_parquet(dst, layout=layout)
    else:
        log.info("saving full dataset")
        lists.to_dataset().save(dst)


def _subset_lists(lists: ItemListCollection, n: int) -> ItemListCollection:
    """
    Subset the user lists.
    """
    log = _log.bind(n_input=len(lists), n_output=n)
    if len(lists) <= n:
        log.warn("input count exceeds target count, keeping all")
        return lists

    rng = random_generator()

    log.info("subsetting lists")
    log.debug("picking row indices")
    rows = rng.choice(len(lists), n, replace=True)

    log.debug("converting to arrow table")
    table = lists.to_arrow()
    log.debug("subsetting arrow table")
    table = table.take(rows)
    log.debug("converting back to item list collection")
    return ItemListCollection.from_arrow(table)
