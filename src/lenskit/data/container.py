# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
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

from lenskit._accel import data as _data_accel
from lenskit.logging import get_logger
from lenskit.logging.stopwatch import Stopwatch

from .schema import DataSchema

_log = get_logger(__name__)


@dataclass(eq=False, order=False)
class DataContainer:
    """
    A general container for the data backing a dataset.
    """

    schema: DataSchema
    tables: dict[str, pa.Table]
    _sorted: bool = False
    _rel_coords: dict[str, _data_accel.CoordinateTable | None] | None = None

    def normalize(self):
        """
        Normalize data to adhere to the expectations of the most current schema.
        """

        self._sort_matrix_relationships()

    def _sort_matrix_relationships(self):
        log = _log.bind(dataset=self.schema.name)
        if self._sorted or self.schema.version >= "2025.3":
            return

        for name, rel in self.schema.relationships.items():
            if len(rel.entities) == 2 and not rel.repeats.is_present:
                # make sure they are a sorted matrix
                tbl = self.tables[name]
                e_cols = [e + "_num" for e in rel.entities.keys()]
                if not _data_accel.is_sorted_coo(tbl.to_batches(), *e_cols):
                    log.debug("sorting non-repeating relationship %s", name)
                    self.tables[name] = tbl.sort_by([(c, "ascending") for c in e_cols])

        self._sorted = True

    def __getstate__(self):
        # work around Ray's broken serialization of extension types
        state = dict(self.__dict__)
        state["tables"] = {name: _make_compat_table(tbl) for name, tbl in self.tables.items()}
        state["_rel_coords"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tables = {name: _resolve_compat_table(tbl) for name, tbl in state["tables"].items()}

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
            tables[name] = read_table(path / f"{name}.parquet").combine_chunks()

        for name in schema.relationships:
            log.debug("reading relationship table %s", name, table=name, time=timer.elapsed())
            tables[name] = read_table(path / f"{name}.parquet").combine_chunks()

        log.debug("finished loading data set in %s", timer, time=timer.elapsed())
        return cls(schema, tables)


def _make_compat_table(tbl: pa.Table):
    schema = tbl.schema
    ftypes = {}

    for name in schema.names:
        field = schema.field(name)
        if isinstance(field.type, pa.BaseExtensionType):
            ftypes[name] = field.type.storage_type
        else:
            ftypes[name] = field.type

    compat_schema = pa.schema(ftypes)
    return schema, tbl.cast(compat_schema, safe=True)


def _resolve_compat_table(data: tuple[pa.Schema, pa.Table]):
    schema, table = data
    return table.cast(schema)
