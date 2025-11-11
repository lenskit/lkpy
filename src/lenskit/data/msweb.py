# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Support for the MSWeb datasets.
"""

import csv
from pathlib import Path
from typing import Literal, overload

import pyarrow as pa
from xopen import xopen

from lenskit.logging import get_logger

from .builder import DatasetBuilder
from .collection import ItemListCollection
from .dataset import Dataset

_log = get_logger(__name__)


@overload
def load_ms_web(path: Path, format: Literal["dataset"] = "dataset") -> Dataset: ...
@overload
def load_ms_web(path: Path, format: Literal["collection"]) -> ItemListCollection: ...
@overload
def load_ms_web(
    path: Path, format: Literal["dataset", "collection"] = "dataset"
) -> Dataset | ItemListCollection: ...
def load_ms_web(
    path: Path, format: Literal["dataset", "collection"] = "dataset"
) -> Dataset | ItemListCollection:
    """
    Load the MSWeb data set.

    The Microsoft Anonymous Web data set was published by
    :cite:t:`breeseEmpiricalAnalysisPredictive1998`, and is available from the
    `UCI repository`_.

    This function can load the data either as a :class:`Dataset` (useful for
    training) or as an :class:`ItemListCollection` (for evaluation).

    .. _UCI repository: https://kdd.ics.uci.edu/databases/msweb/msweb.html

    Args:
        path:
            The path to the data file (gzip-compressed).
        format:
            The type of object to load the data set into.
    Returns:
        The loaded MSWeb data.
    """
    ds = _load_ms_dataset(path)
    match format:
        case "collection":
            return ds.interactions("visit").item_lists()
        case "dataset":
            return ds
        case _:  # pragma: nocover
            raise ValueError(f"invalid format: {format}")


def _load_ms_dataset(path: Path) -> Dataset:
    item_ids = []
    item_titles = []
    item_urls = []
    session_votes = []
    cur_session = None
    _log.info("opening MSWeb file", path=str(path))
    with xopen(path, "rt") as data:
        reader = csv.reader(data)
        for row in reader:
            code = row[0]
            match code:
                case "A":
                    _c, vid, _n, title, url = row
                    item_ids.append(int(vid))
                    item_titles.append(title)
                    item_urls.append(url)
                case "C":
                    _c, _sname, sid = row
                    cur_session = int(sid)
                case "V":
                    _c, vid, _n = row
                    session_votes.append({"session_id": cur_session, "item_id": int(vid)})

    dsb = DatasetBuilder("ms-web")
    dsb.add_entities("item", item_ids)
    dsb.add_scalar_attribute("item", "title", item_ids, item_titles)
    dsb.add_scalar_attribute("item", "url", item_ids, item_urls)
    dsb.add_entity_class("session")
    votes = pa.Table.from_pylist(session_votes)
    dsb.add_interactions(
        "visit",
        votes,
        entities=["session", "item"],
        missing="insert",
        default=True,
        allow_repeats=False,
    )
    return dsb.build()
