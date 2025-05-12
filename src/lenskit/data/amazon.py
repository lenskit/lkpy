# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Load Amazon ratings data from Julian McAuley's group.
"""

import re
from collections.abc import Generator
from io import BufferedReader
from os import PathLike
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.csv import ConvertOptions, ReadOptions, open_csv
from xopen import xopen

from lenskit.logging import get_logger

from .builder import DatasetBuilder
from .dataset import Dataset

_log = get_logger(__name__)


def load_amazon_ratings(*files: Path | str | PathLike[str]) -> Dataset:
    """
    Load an Amazon ratings CSV file into a dataset.  Files may be compressed by
    any compressor supported by :mod:`xopen` and PyArrow.

    The era of data (2014, 2018, or 2023) is auto-detected from file content.

    Args:
        files:
            The source rating files. Each one will be appeded in turn to the
            loader.
    """

    dsb = DatasetBuilder()
    dsb.add_entity_class("user")
    dsb.add_relationship_class("rating", ["user", "item"], allow_repeats=False, interaction=True)

    version = "AZ?"
    category = "Unknown"

    for file in files:
        file = Path(file)
        log = _log.bind(file=str(file))
        log.debug("opening file")

        with xopen(file, "rb", threads=1) as xf, BufferedReader(xf) as bf:  # type: ignore
            block = bf.peek(4096)
            log.debug("file header: %r", block[:20])
            if block.startswith(b"user_id,parent_asin,"):
                reader = open_az_2023(bf)
                version = "AZ23"
            elif block[:1] == b"A":
                reader = open_az_2014(bf)
                version = "AZ14"
            else:
                reader = open_az_2018(bf)
                version = "AZ18"

            m = re.match(r"^(?:ratings_)?(.*?)\.", file.name)
            if m is not None:
                category = m[1]

            # update the name appropriately
            name = f"{version}-{category}"
            if dsb.name is None:
                dsb.schema.name = name
            elif dsb.name != name:
                dsb.schema.name = f"{version}-Mixed"

            log.info("reading %s %s", version, category)

            for tbl in reader:
                dsb.add_interactions("rating", tbl, missing="insert")

    return dsb.build()


def open_az_2023(input: BinaryIO) -> Generator[pa.Table]:
    # 2023 Amazon data: has labels, we will ignore history
    with open_csv(
        input,
        convert_options=ConvertOptions(
            include_columns=["user_id", "parent_asin", "rating", "timestamp"],
            column_types={"rating": pa.float32()},
        ),
    ) as reader:
        for batch in reader:
            batch = batch.rename_columns(["user_id", "item_id", "rating", "timestamp"])
            columns = {c: batch.column(c) for c in batch.column_names}
            columns["timestamp"] = pc.cast(batch.column("timestamp"), pa.timestamp("ms"))
            yield pa.Table.from_pydict(columns)


def open_az_2014(input: BinaryIO) -> Generator[pa.Table]:
    # 2014 Amazon data: user, item, rating, timestamp
    with open_csv(
        input,
        read_options=ReadOptions(column_names=["user_id", "item_id", "rating", "timestamp"]),
        convert_options=ConvertOptions(
            column_types={"rating": pa.float32()},
        ),
    ) as reader:
        for batch in reader:
            columns = {c: batch.column(c) for c in batch.column_names}
            columns["timestamp"] = pc.cast(batch.column("timestamp"), pa.timestamp("s"))
            yield pa.Table.from_pydict(columns)


def open_az_2018(input: BinaryIO) -> Generator[pa.Table]:
    # 2018 Amazon data: user, item, rating, timestamp
    with open_csv(
        input,
        read_options=ReadOptions(
            block_size=16 * 1024 * 1024,
            column_names=["item_id", "user_id", "rating", "timestamp"],
        ),
        convert_options=ConvertOptions(
            column_types={"rating": pa.float32()},
        ),
    ) as reader:
        for batch in reader:
            columns = {c: batch.column(c) for c in batch.column_names}
            columns["timestamp"] = pc.cast(batch.column("timestamp"), pa.timestamp("s"))
            yield pa.Table.from_pydict(columns)
