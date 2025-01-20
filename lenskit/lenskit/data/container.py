"""
Data containers, the internal storage of data sets.
"""

from dataclasses import dataclass

import pyarrow as pa

from .schema import DataSchema


@dataclass(eq=False, order=False)
class DataContainer:
    schema: DataSchema
    tables: dict[str, pa.Table]
