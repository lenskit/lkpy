# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Data containers, the internal storage of data sets.
"""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from shutil import rmtree

import pyarrow as pa
from pyarrow.parquet import read_table, write_table

from lenskit.logging import get_logger
from lenskit.logging.stopwatch import Stopwatch

from .schema import DataSchema

_log = get_logger(__name__)


@dataclass(eq=False, order=False)
class DataContainer:
    schema: DataSchema
    tables: dict[str, pa.Table]

    def save(self, path: str | PathLike[str]):
        """
        Save the data to disk.
        """
        from .summary import save_stats

        path = Path(path)
        log = _log.bind(name=self.schema.name, path=str(path))
        log.info("saving dataset")

        if path.exists():
            log.warn("path already exists, removing")
            rmtree(path)

        log.debug("ensuring path exists")
        path.mkdir(exist_ok=True, parents=True)

        log.debug("writing schema")
        with open(path / "schema.json", "wt") as jsf:
            print(self.schema.model_dump_json(), file=jsf)

        for name, table in self.tables.items():
            log.debug("writing table", table=name, rows=table.num_rows)
            write_table(table, path / f"{name}.parquet", compression="zstd")

        log.debug("writing summary file")
        save_stats(self, path / "summary.md")

    @classmethod
    def load(cls, path: str | PathLike[str]):
        """
        Load data from disk.
        """
        path = Path(path)
        log = _log.bind(path=str(path))

        timer = Stopwatch()
        log.info("loading dataset")
        log.debug("reading schema")
        schema_file = path / "schema.json"
        schema = DataSchema.model_validate_json(schema_file.read_text(encoding="utf8"))
        log = log.bind(name=schema.name)

        tables = {}
        for name in schema.entities:
            log.debug("reading entity table %s", name, table=name, time=timer.elapsed())
            tables[name] = read_table(path / f"{name}.parquet")

        for name in schema.relationships:
            log.debug("reading relationship table %s", name, table=name, time=timer.elapsed())
            tables[name] = read_table(path / f"{name}.parquet")

        log.debug("finished loading data set in %s", timer, time=timer.elapsed())
        return cls(schema, tables)
