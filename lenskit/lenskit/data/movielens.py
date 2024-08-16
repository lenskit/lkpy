# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Code to import MovieLens data sets into LensKit.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeAlias
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .convert import from_interactions_df
from .dataset import Dataset

_log = logging.getLogger(__name__)

LOC: TypeAlias = Path | tuple[ZipFile, str]


class MLVersion(Enum):
    ML_100K = "ml-100k"
    ML_1M = "ml-1m"
    ML_10M = "ml-10m"
    ML_20M = "ml-20m"
    ML_25M = "ml-25m"
    ML_LATEST_SMALL = "ml-latest-small"
    ML_LATEST = "ml-latest"
    ML_MODERN = "ml-modern"


@dataclass
class MLData:
    version: MLVersion
    source: Path | ZipFile
    prefix: str = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if isinstance(self.source, ZipFile):
            self.source.close()

    def open_file(self, name: str):
        if isinstance(self.source, Path):
            return open(self.source / (self.prefix + name), "r")
        else:
            return self.source.open(self.prefix + name)


def load_movielens(path: str | Path) -> Dataset:
    """
    Load a MovieLens dataset.  The appropriate MovieLens format is detected
    based on the file contents.

    Args:
        path:
            The path to the dataset, either as an unpacked directory or a zip
            file.

    Returns:
        The dataset.
    """
    df = load_movielens_df(path)
    return from_interactions_df(df)


def load_movielens_df(path: str | Path) -> pd.DataFrame:
    """
    Load the ratings from a MovieLens dataset as a raw data frame.  The
    appropriate MovieLens format is detected based on the file contents.

    Args:
        path:
            The path to the dataset, either as an unpacked directory or a zip
            file.

    Returns:
        The ratings, with columns ``user``, ``item``, ``rating``, and
        ``timestamp``.
    """
    with _ml_detect_and_open(path) as ml:
        match ml.version:
            case MLVersion.ML_100K:
                return _load_ml_100k(ml)
            case MLVersion.ML_1M | MLVersion.ML_10M:
                return _load_ml_million(ml)
            case _:
                return _load_ml_modern(ml)


def _ml_detect_and_open(path: str | Path) -> MLData:
    loc = Path(path)
    ds: MLVersion

    if loc.is_file() and loc.suffix == ".zip":
        _log.debug("opening zip file at %s", loc)
        zf = ZipFile(loc, "r")
        try:
            infos = zf.infolist()
            first = infos[0]
            if not first.is_dir:
                _log.error("%s: first entry is not directory")
                raise RuntimeError("invalid ML zip file")

            _log.debug("%s: base dir filename %s", loc, first.filename)
            dsm = re.match(r"^(ml-\d+[MmKk])", first.filename)
            if not dsm:
                _log.error("%s: invalid directory name %s", loc, first.filename)
                raise RuntimeError("invalid ML zip file")

            ds = MLVersion(dsm.group(1).lower())
            _log.debug("%s: found ML data set %s", loc, ds)
            return MLData(ds, zf, first.filename)
        except Exception as e:  # pragma nocover
            zf.close()
            raise e
    else:
        _log.debug("loading from directory %s", loc)
        dsm = re.match(r"^(ml-\d+[MmKk])", loc.name)
        if dsm:
            ds = MLVersion(dsm.group(1))
            _log.debug("%s: inferred data set %s from dir name", loc, ds)
        else:
            _log.debug("%s: checking contents for data type", loc)
            if (loc / "u.data").exists():
                _log.debug("%s: found u.data, interpreting as 100K")
                ds = MLVersion.ML_100K
            elif (loc / "ratings.dat").exists():
                if (loc / "tags.dat").exists():
                    _log.debug("%s: found ratings.dat and tags.dat, interpreting as 10M", loc)
                    ds = MLVersion.ML_10M
                else:
                    _log.debug("%s: found ratings.dat but no tags, interpreting as 1M", loc)
                    ds = MLVersion.ML_1M
            elif (loc / "ratings.csv").exists():
                _log.debug("%s: found ratings.csv, interpreting as modern (20M and later)", loc)
                ds = MLVersion.ML_MODERN
            else:
                _log.error("%s: could not detect MovieLens data", loc)
                raise RuntimeError("invalid ML directory")

        return MLData(ds, loc)


def _load_ml_100k(ml: MLData) -> pd.DataFrame:
    with ml.open_file("u.data") as data:
        return pd.read_csv(
            data,
            sep="\t",
            header=None,
            names=["user", "item", "rating", "timestamp"],
            dtype={
                "user": np.int32,
                "item": np.int32,
                "rating": np.float32,
                "timestamp": np.int32,
            },
        )


def _load_ml_million(ml: MLData) -> pd.DataFrame:
    with ml.open_file("ratings.dat") as data:
        return pd.read_csv(
            data,
            sep=":",
            header=None,
            names=["user", "_ui", "item", "_ir", "rating", "_rt", "timestamp"],
            usecols=[0, 2, 4, 6],
            dtype={
                "user": np.int32,
                "item": np.int32,
                "rating": np.float32,
                "timestamp": np.int32,
            },
        )


def _load_ml_modern(ml: MLData) -> pd.DataFrame:
    with ml.open_file("ratings.csv") as data:
        return pd.read_csv(
            data,
            dtype={
                "userId": np.int32,
                "movieId": np.int32,
                "rating": np.float32,
                "timestamp": np.int64,
            },
        ).rename(columns={"userId": "user", "movieId": "item"})
