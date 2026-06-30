# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Code for saving Parquet item list collections.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Generic, Literal

import pyarrow as pa
from pyarrow.parquet import ParquetWriter

from lenskit.data import ListILC
from lenskit.logging import get_logger

from .._items import ItemList
from ._base import ItemListCollector
from ._keys import ID, K

_log = get_logger(__name__)


class ParquetItemListCollector(ItemListCollector, Generic[K]):
    """
    Item list collector that saves lists by batches to a Parquet file
    (in native format).
    """

    batch_size: int
    path: Path
    writer: ParquetWriter | None = None
    compression: Literal["zstd", "snappy"] | None
    _cur_batch: ListILC[K]

    def __init__(
        self,
        path: Path | os.PathLike[str],
        key: type[K] | Sequence[str],
        batch_size: int = 5000,
        compression: Literal["zstd", "snappy"] | None = "zstd",
    ):
        self.batch_size = batch_size
        self.path = Path(path)
        self.compression = compression
        self._cur_batch = ListILC(key, index=False)

    def add(self, list: ItemList, *fields: ID, **kwfields: ID):
        self._cur_batch.add(list, *fields, **kwfields)
        self._maybe_flush()

    def close(self):
        self._flush()
        assert self.writer is not None
        self.writer.close()

    def _maybe_flush(self):
        if len(self._cur_batch) >= self.batch_size:
            self._flush()

    def _flush(self):
        for batch in self._cur_batch.record_batches(self.batch_size):
            if self.writer is None:
                _log.debug("opening Parquet writer", schema=batch.schema, file=str(self.path))
                self.writer = ParquetWriter(
                    self.path, batch.schema, compression=self.compression or "none"
                )
            self.writer.write_batch(batch)

        if self.writer is None:
            _log.warning("creating empty writer", file=str(self.path))
            schema = {k: pa.null() for k in self._cur_batch.key_fields}
            schema["item_id"] = pa.null()
            self.writer = ParquetWriter(
                self.path, pa.schema(schema), compression=self.compression or "none"
            )
