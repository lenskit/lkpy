"""
Generate summaries of datasets.
"""

from os import PathLike
from pathlib import Path
from typing import TextIO

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


def _write_stats(data: DataContainer, stats: TextIO):
    if data.schema.name:
        print("# Summary of", data.schema.name, file=stats)
    else:
        print("# Summary of Unnamed Data", file=stats)

    print(file=stats)
    print("## Entities\n", file=stats)
    for name, schema in data.schema.entities.items():
        print(
            "- {} ({}, {} attributes)".format(
                name, metric(data.tables[name].num_rows), len(schema.attributes)
            ),
            file=stats,
        )
    print(file=stats)

    print("## Relationships\n", file=stats)
    for name, schema in data.schema.relationships.items():
        print(
            "- {} ({}, {} attributes{})".format(
                name,
                metric(data.tables[name].num_rows),
                len(schema.attributes),
                ", interaction" if schema.interaction else "",
            ),
            file=stats,
        )

    print(file=stats)

    print("## Data Tables\n", file=stats)
    print(table_stats(data), file=stats)


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
