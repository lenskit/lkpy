# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Load Amazon ratings data from Julian McAuley's group.
"""

import re
from io import BufferedReader
from os import PathLike
from pathlib import Path

import pyarrow as pa
from pyarrow.csv import ConvertOptions, ReadOptions, open_csv
from xopen import xopen

from lenskit.logging import get_logger

from .builder import DatasetBuilder
from .dataset import Dataset

_log = get_logger(__name__)


def load_amazon_ratings(file: Path | str | PathLike[str]) -> Dataset:
    """
    Load an Amazon ratings CSV file into a dataset.  Files may be compressed by
    any compressor supported by :mod:`xopen` and PyArrow.

    The era of data (2014, 2018, or 2023) is auto-detected from file content.
    """
    file = Path(file)
    category = "Unknown"

    log = _log.bind(file=str(file))
    log.debug("opening file")
    with xopen(file, "rb", threads=1) as xf, BufferedReader(xf) as bf:
        block = bf.peek(4096)
        log.debug("file header: %r", block[:20])
        if block.startswith(b"user_id,parent_asin,"):
            # 2023 Amazon data with headers
            reader = open_csv(
                bf,
                convert_options=ConvertOptions(
                    include_columns=["user_id", "parent_asin", "rating", "timestamp"],
                    column_types={"rating": pa.float32()},
                ),
            )
            version = "AZ23"
            m = re.match(r"^(?:ratings_)?(.*?)\.", file.name)
            if m is not None:
                category = m[1]
        elif block[:1] == b"A":
            # 2014 Amazon data: user, item, rating, timestamp
            reader = open_csv(
                bf,
                read_options=ReadOptions(
                    column_names=["user_id", "item_id", "rating", "timestamp"]
                ),
                convert_options=ConvertOptions(
                    column_types={"rating": pa.float32()},
                ),
            )
            version = "AZ14"
            m = re.match(r"^(?:ratings_)?(.*?)\.", file.name)
            if m is not None:
                category = m[1]
        else:
            # 2018 Amazon data: user, item, rating, timestamp
            reader = open_csv(
                bf,
                read_options=ReadOptions(
                    block_size=16 * 1024 * 1024,
                    column_names=["item_id", "user_id", "rating", "timestamp"],
                ),
                convert_options=ConvertOptions(
                    column_types={"rating": pa.float32()},
                ),
            )
            version = "AZ18"
            m = re.match(r"^(?:ratings_)?(.*?)\.", file.name)
            if m is not None:
                category = m[1]

        log.info("reading %s %s", version, category)
        dsb = DatasetBuilder(f"{version}-{category}")
        dsb.add_entity_class("user")
        dsb.add_relationship_class(
            "rating", ["user", "item"], allow_repeats=False, interaction=True
        )

        for batch in reader:
            tbl = pa.Table.from_batches([batch])
            tbl = tbl.rename_columns(
                ["item_id" if c == "parent_asin" else c for c in tbl.column_names]
            )
            dsb.add_interactions("rating", tbl, missing="insert")

    return dsb.build()
