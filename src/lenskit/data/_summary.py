# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Generate summaries of datasets.
"""

from os import PathLike
from pathlib import Path
from typing import TextIO

import pyarrow as pa
import pyarrow.compute as pc
from humanize import metric, naturalsize
from prettytable import PrettyTable, TableStyle
from structlog.stdlib import BoundLogger

from lenskit.logging import get_logger

from ._container import DataContainer
from ._dataset import Dataset
from .matrix import SparseRowType
from .schema import AttrLayout, ColumnSpec

_log = get_logger(__name__)


def save_stats(data: Dataset | DataContainer, out: str | Path | PathLike[str] | TextIO):
    """
    Save dataset statistics to a file or output stream.
    """
    if isinstance(data, Dataset):
        data = data._data

    log = _log.bind()

    if hasattr(out, "write"):
        _write_stats(data, out, log)  # type: ignore
    else:
        log = log.bind(file=str(out))
        with open(out, "wt") as stats:
            _write_stats(data, stats, log)


def _write_stats(data: DataContainer, out: TextIO, log: BoundLogger):
    if data.schema.name:
        print("# Summary of", data.schema.name, file=out)
    else:
        print("# Summary of Unnamed Data", file=out)
    print(file=out)

    _write_entities(data, out, log)
    _write_relationships(data, out, log)

    print("## Data Tables\n", file=out)
    print(table_stats(data), file=out)


def _write_entities(data: DataContainer, out: TextIO, log: BoundLogger):
    print("## Entities\n", file=out)
    log.debug("summarizing entities")
    for name, schema in data.schema.entities.items():
        print(
            f"- {name} ({metric(data.tables[name].num_rows)}, {len(schema.attributes)} attributes)",
            file=out,
        )
    print(file=out)

    for name in data.schema.entities:
        _write_entity_info(data, name, out, log)


def _write_entity_info(data: DataContainer, name: str, out: TextIO, log: BoundLogger):
    tbl = data.tables[name]
    log = log.bind(entity=name)
    log.debug("describing entity")
    print(f"### Entity `{name}`\n", file=out)
    print(f"- {tbl.num_rows:,d} instances\n", file=out)

    attributes = data.schema.entities[name].attributes
    if attributes:
        print("#### Attributes\n", file=out)
        print(_attr_table(tbl, attributes), file=out)
        print(file=out)

    print("#### Schema\n", file=out)
    print("```", file=out)
    print(tbl.schema.to_string(), file=out)
    print("```\n", file=out)


def _write_relationships(data: DataContainer, out: TextIO, log: BoundLogger):
    print("## Relationships\n", file=out)
    log.debug("summarizing relationships")
    for name, schema in data.schema.relationships.items():
        print(
            "- {} ({}, {} attributes{})".format(
                name,
                metric(data.tables[name].num_rows),
                len(schema.attributes),
                ", interaction" if schema.interaction else "",
            ),
            file=out,
        )
    print(file=out)

    for name in data.schema.relationships:
        _write_relationship_info(data, name, out, log)


def _write_relationship_info(data: DataContainer, name: str, out: TextIO, log: BoundLogger):
    tbl = data.tables[name]
    log = log.bind(relationship=name)
    log.debug("describing relationship")
    print(f"### Relationship `{name}`\n", file=out)
    print(f"- {tbl.num_rows:,d} records\n", file=out)

    print("#### Entities\n", file=out)
    pt = PrettyTable()
    pt.set_style(TableStyle.MARKDOWN)
    pt.field_names = ["Name", "Class", "Unique Count"]
    pt.align["Name"] = "l"
    pt.align["Unique Count"] = "r"
    pt.custom_format["Unique Count"] = lambda _, v: f"{v:,d}"
    for e_name, e_cls in data.schema.relationships[name].entities.items():
        e_cls = e_cls or e_name
        e_col = tbl.column(e_name + "_num")
        pt.add_row([e_name, e_cls, pc.count_distinct(e_col).as_py()])
    print(pt, file=out)
    print(file=out)

    attributes = data.schema.relationships[name].attributes
    if attributes:
        print("#### Attributes\n", file=out)
        print(_attr_table(tbl, attributes), file=out)
        print(file=out)


def _attr_table(tbl: pa.Table, attributes: dict[str, ColumnSpec]):
    pt = PrettyTable()
    pt.set_style(TableStyle.MARKDOWN)
    pt.field_names = ["Name", "Layout", "Type", "Dimension", "Count", "Size"]
    pt.align["Name"] = "l"
    pt.align["Dimension"] = "r"
    pt.align["Count"] = "r"
    pt.custom_format["Count"] = lambda _, v: f"{v:,d}"
    pt.align["Size"] = "r"
    pt.custom_format["Size"] = lambda _, v: naturalsize(v, binary=True)
    for a_name, a_schema in attributes.items():
        field = tbl.field(a_name)
        col = tbl.column(a_name)
        vec_type = field.type
        dim = "-"
        match a_schema.layout:
            case AttrLayout.LIST:
                assert isinstance(vec_type, pa.ListType)
                vec_type = vec_type.value_type
            case AttrLayout.VECTOR:
                assert isinstance(vec_type, (pa.ListType, pa.FixedSizeListType))
                vec_type = vec_type.value_type
                dim = f"{a_schema.vector_size:,d}"
            case AttrLayout.SPARSE:
                assert isinstance(vec_type, SparseRowType)
                vec_type = vec_type.value_type
                dim = f"{a_schema.vector_size:,d}"

        pt.add_row(
            [
                a_name,
                a_schema.layout.value,
                vec_type,
                dim,
                tbl.num_rows - col.null_count,
                col.nbytes,
            ]
        )

    return pt


def table_stats(data: DataContainer) -> PrettyTable:
    tbl = PrettyTable()
    tbl.set_style(TableStyle.MARKDOWN)
    tbl.field_names = ["Name", "Rows", "Bytes"]
    tbl.align["Name"] = "l"
    tbl.align["Rows"] = "r"
    tbl.custom_format["Rows"] = lambda _, v: f"{v:,d}"
    tbl.align["Bytes"] = "r"
    tbl.custom_format["Bytes"] = lambda _, v: naturalsize(v, binary=True)

    for name, table in data.tables.items():
        tbl.add_row([name, table.num_rows, table.nbytes])

    return tbl
