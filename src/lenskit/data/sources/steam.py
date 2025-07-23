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
                    pa.field("playtime_2weeks", pa.string()),
                    pa.field("playtime_forever", pa.string()),
                ]
            )
        ),
    }
)


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
