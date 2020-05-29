"""
Data utilities
"""

import logging
import pathlib

try:
    import fastparquet
except ImportError:
    fastparquet = None

_log = logging.getLogger(__name__)


def read_df_detect(path):
    """
    Read a Pandas data frame, auto-detecting the file format based on filename suffix.
    The following file types are supported:

    CSV
        File has suffix ``.csv``, read with :py:func:`pandas.read_csv`.
    Parquet
        File has suffix ``.parquet``, ``.parq``, or ``.pq``, read with
        :py:func:`pandas.read_parquet`.
    """
    import pandas as pd
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix in ('.parquet', '.parq', '.pq'):
        return pd.read_parquet(path)
