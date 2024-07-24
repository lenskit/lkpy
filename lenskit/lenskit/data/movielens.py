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
from pathlib import Path
from typing import TypeAlias
from zipfile import ZipFile

import numpy as np
import pandas as pd

from lenskit.data.dataset import Dataset, from_interactions_df

_log = logging.getLogger(__name__)

LOC: TypeAlias = Path | tuple[ZipFile, str]


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
    loc = Path(path)
    if loc.is_file() and loc.suffix == ".zip":
        _log.debug("opening zip file at %s", loc)
        with ZipFile(loc, "r") as zf:
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

            ds = dsm.group(1).lower()
            _log.debug("%s: found ML data set %s", loc, ds)
            return _load_for_type((zf, first.filename), ds)
    else:
        _log.debug("loading from directory %s", loc)
        dsm = re.match(r"^(ml-\d+[MmKk])", loc.name)
        if dsm:
            ds = dsm.group(1)
            _log.debug("%s: inferred data set %s from dir name", loc, ds)
        else:
            _log.debug("%s: checking contents for data type", loc)
            if (loc / "u.data").exists():
                _log.debug("%s: found u.data, interpreting as 100K")
                ds = "ml-100k"
            elif (loc / "ratings.dat").exists():
                if (loc / "tags.dat").exists():
                    _log.debug("%s: found ratings.dat and tags.dat, interpreting as 10M", loc)
                    ds = "ml-10m"
                else:
                    _log.debug("%s: found ratings.dat but no tags, interpreting as 1M", loc)
                    ds = "ml-1m"
            elif (loc / "ratings.csv").exists():
                _log.debug("%s: found ratings.csv, interpreting as modern (20M and later)", loc)
                ds = "ml-modern"
            else:
                _log.error("%s: could not detect MovieLens data", loc)
                raise RuntimeError("invalid ML directory")

        return _load_for_type(loc, ds)


def _load_for_type(loc: LOC, ds: str) -> Dataset:
    "Load the specified MovieLens data set"
    match ds:
        case "ml-100k":
            return _load_ml_100k(loc)
        case "ml-1m" | "ml-10m":
            return _load_ml_million(loc)
        case _:
            return _load_ml_modern(loc)


def _load_ml_100k(loc: LOC) -> Dataset:
    with _open_file(loc, "u.data") as data:
        rates_df = pd.read_csv(
            data,
            sep="\t",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
            dtype={
                "user_id": np.int32,
                "item_id": np.int32,
                "rating": np.float32,
                "timestamp": np.int32,
            },
        )

    return from_interactions_df(rates_df)


def _load_ml_million(loc: LOC) -> Dataset:
    with _open_file(loc, "ratings.dat") as data:
        rates_df = pd.read_csv(
            data,
            sep=":",
            header=None,
            names=["user_id", "_ui", "item_id", "_ir", "rating", "_rt", "timestamp"],
            usecols=[0, 2, 4, 6],
            dtype={
                "user_id": np.int32,
                "item_id": np.int32,
                "rating": np.float32,
                "timestamp": np.int32,
            },
        )

    return from_interactions_df(rates_df)


def _load_ml_modern(loc: LOC) -> Dataset:
    with _open_file(loc, "ratings.csv") as data:
        rates_df = pd.read_csv(
            data,
            dtype={
                "userId": np.int32,
                "movieId": np.int32,
                "rating": np.float32,
                "timestamp": np.int64,
            },
        )

    return from_interactions_df(rates_df, item_col="movieId")


def _open_file(loc: LOC, name: str):
    if isinstance(loc, Path):
        return open(loc / name, "r")
    else:
        zf, root = loc
        return zf.open(root + name)
