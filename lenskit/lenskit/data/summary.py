"""
Generate summaries of datasets.
"""

from os import PathLike
from pathlib import Path
from typing import TextIO

import pyarrow.compute as pc
from humanize import metric, naturalsize
from prettytable import PrettyTable, TableStyle

from .container import DataContainer
from .dataset import Dataset


def save_stats(data: Dataset | DataContainer, out: str | Path | PathLike[str] | TextIO):
    """
    Save dataset statistics to a file or output stream.
    """

    if isinstance(data, Dataset):
        data = data._data

    if hasattr(out, "write"):
        _write_stats(data, out)  # type: ignore
    else:
        with open(out, "wt") as stats:  # type: ignore
            _write_stats(data, stats)


def _write_stats(data: DataContainer, out: TextIO):
    if data.schema.name:
        print("# Summary of", data.schema.name, file=out)
    else:
        print("# Summary of Unnamed Data", file=out)
    print(file=out)

    _write_entities(data, out)
    _write_relationships(data, out)

    print("## Data Tables\n", file=out)
    print(table_stats(data), file=out)


def _write_entities(data: DataContainer, out: TextIO):
    print("## Entities\n", file=out)
    for name, schema in data.schema.entities.items():
        print(
            "- {} ({}, {} attributes)".format(
                name, metric(data.tables[name].num_rows), len(schema.attributes)
            ),
            file=out,
        )
    print(file=out)

    for name in data.schema.entities.keys():
        _write_entity_info(data, name, out)


def _write_entity_info(data: DataContainer, name: str, out: TextIO):
    tbl = data.tables[name]
    print(f"### Entity `{name}`\n", file=out)
    print("- {:,d} instances\n".format(tbl.num_rows), file=out)

    pt = PrettyTable()
    pt.set_style(TableStyle.MARKDOWN)
    pt.field_names = ["Name", "Layout", "Count"]
    pt.align["Name"] = "l"
    pt.align["Count"] = "r"
    pt.int_format = "{:,d}"
    for a_name, a_schema in data.schema.entities[name].attributes.items():
        pt.add_row([a_name, a_schema.layout, tbl.num_rows])

    print(pt, file=out)
    print(file=out)


def _write_relationships(data: DataContainer, out: TextIO):
    print("## Relationships\n", file=out)
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

    for name in data.schema.relationships.keys():
        _write_relationship_info(data, name, out)


def _write_relationship_info(data: DataContainer, name: str, out: TextIO):
    tbl = data.tables[name]
    print(f"### Relationship `{name}`\n", file=out)
    print("- {:,d} records\n".format(tbl.num_rows), file=out)

    print("#### Entities\n", file=out)
    pt = PrettyTable()
    pt.set_style(TableStyle.MARKDOWN)
    pt.field_names = ["Name", "Class", "Unique Count"]
    pt.int_format = "{:,d}"
    pt.align["Name"] = "l"
    pt.align["Unique Count"] = "r"
    for e_name, e_cls in data.schema.relationships[name].entities.items():
        e_cls = e_cls or e_name
        e_col = tbl.column(e_name)
        pt.add_row([e_name, e_cls, pc.count_distinct(e_col).as_py()])

    print("#### Attributes\n", file=out)
    pt = PrettyTable()
    pt.set_style(TableStyle.MARKDOWN)
    pt.field_names = ["Name", "Layout", "Count"]
    pt.align["Name"] = "l"
    pt.align["Count"] = "r"
    pt.int_format = "{:,d}"
    for a_name, a_schema in data.schema.relationships[name].attributes.items():
        pt.add_row([a_name, a_schema.layout, tbl.num_rows])

    print(pt, file=out)
    print(file=out)


def table_stats(data: DataContainer) -> PrettyTable:
    tbl = PrettyTable()
    tbl.set_style(TableStyle.MARKDOWN)
    tbl.field_names = ["Name", "Rows", "Bytes"]
    tbl.align["Name"] = "l"
    tbl.align["Rows"] = "r"
    tbl.custom_format["Rows"] = lambda _, v: "{:,d}".format(v)
    tbl.align["Bytes"] = "r"
    tbl.custom_format["Bytes"] = lambda _, v: naturalsize(v, binary=True)

    for name, table in data.tables.items():
        tbl.add_row([name, table.num_rows, table.nbytes])

    return tbl
