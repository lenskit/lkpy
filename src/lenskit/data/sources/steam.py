# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Support for importing `Steam data`_.

.. _Steam data: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data
"""

import sys
from ast import literal_eval
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pyarrow as pa
from more_itertools import chunked
from xopen import xopen

from lenskit.data import Dataset, DatasetBuilder
from lenskit.diagnostics import DataError
from lenskit.logging import get_logger

_log = get_logger(__name__)

BATCH_SIZE = 10_000
AU_USERS_ITEMS_SCHEMA = pa.schema(
    {
        "user_id": pa.string(),
        "items_count": pa.int32(),
        "steam_id": pa.string(),
        "user_url": pa.string(),
        "items": pa.list_(
            pa.struct(
                [
                    pa.field("item_id", pa.string()),
                    pa.field("item_name", pa.string()),
                    pa.field("playtime_2weeks", pa.float32()),
                    pa.field("playtime_forever", pa.float32()),
                ]
            )
        ),
    }
)


def load_steam(*files: Path, reviews: bool = False) -> Dataset:
    """
    Load a `Steam dataset`_ from Julian McAuley's group at UCSD.

    .. _Steam dataset: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data

    .. note::

        This function uses filenames to detect which version of the data to
        load, so the files should be named exactly as they are from McAuley's
        download site, optionally recompressed.

    Args:
        files:
            Input files to read, in any order.  Can only load one version of
            the data at a time (Australian or full).
        reviews:
            Set to ``True`` to include review text in the loaded dataset.
    """
    au_interactions = None
    au_reviews = None
    all_games = None
    all_reviews = None

    # autodetect our file(s)
    for file in files:
        name = file.name
        if name.startswith("australian_user_reviews.json"):
            au_reviews = file
        elif name.startswith("australian_users_items.json"):
            au_interactions = file
        elif name.startswith("steam_games.json"):
            all_games = file
        elif name.startswith("steam_reviews.json"):
            all_reviews = file

    if au_interactions is not None:
        _log.debug("looking for Australian subset interactions")
        if all_reviews is not None or all_games is not None:
            _log.error("cannot specify both Australian and overall input files")
            raise DataError("invalid combination of Steam input files")

        return _load_au_steam(au_interactions, au_reviews if reviews else None)

    elif all_reviews is not None:
        _log.debug("looking for full-data inteactions")
        if au_reviews is not None:
            _log.error("cannot specify both Australian and overall input files")
            raise DataError("invalid combination of Steam input files")

        return _load_all_steam(all_games, all_reviews, include_reviews=reviews)
    else:
        _log.error("must supply one of australian_users_items or steam_reviews")
        raise DataError("no Steam interactions provided")


def _load_au_steam(interactions: Path, reviews: Path | None) -> Dataset:
    _dsb = DatasetBuilder()

    _ui_data = _read_table(interactions, AU_USERS_ITEMS_SCHEMA)


def _load_all_steam(games: Path | None, reviews: Path, *, include_reviews: bool) -> Dataset:
    raise NotImplementedError()


def _read_table(path: Path, schema: pa.Schema | None = None) -> pa.Table:
    _log.debug("reading table from loose JSON", file=str(path))
    tbl = pa.Table.from_batches(_decode_chunks(path, schema))
    _log.debug("finished reading table", rows=tbl.num_rows, file=str(path))
    return tbl


def _decode_chunks(path: Path, schema: pa.Schema | None = None) -> Generator[pa.RecordBatch]:
    for chunk in chunked(_decode_steam(path), BATCH_SIZE):
        batch = pa.RecordBatch.from_pylist(chunk, schema)
        if schema is None:
            schema = batch.schema
        yield batch


def _decode_steam(path: Path) -> Generator[dict[str, Any]]:
    """
    Decode a stream of malformed JSON.
    """
    with xopen(path, "rt") as stream:
        for line in stream:
            yield literal_eval(line)


def _preview_file(path: str | Path):
    path = Path(path)
    batch = next(_decode_chunks(path))
    print(batch.schema)


if __name__ == "__main__":
    _preview_file(sys.argv[1])
