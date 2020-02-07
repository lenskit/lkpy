"""
Data utilities
"""

import os
import os.path
import logging
import pathlib
import warnings

import pandas as pd

from ..datasets import MovieLens

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
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix in ('.parquet', '.parq', '.pq'):
        return pd.read_parquet(path)


def load_ml_ratings(path='ml-latest-small'):
    """
    Load the ratings from a modern MovieLens data set (ML-20M or one of the ‘latest’ data sets).

    >>> load_ml_ratings().head()
        user item rating  timestamp
    0   1      31    2.5 1260759144
    1   1    1029    3.0 1260759179
    2   1    1061    3.0 1260759182
    3   1    1129    2.0 1260759185
    4   1    1172    4.0 1260759205

    .. note:: This is deprecated; use :cls:`lenskit.datasets.MovieLens`.

    Args:
        path: The path where the MovieLens data is unpacked.

    Returns:
        pandas.DataFrame:
            The rating data, with user and item columns named properly for LensKit.
    """
    warnings.warn('load_ml_ratings is deprecated, use datsets.MovieLens.')

    ml = MovieLens(path)
    return ml.ratings
