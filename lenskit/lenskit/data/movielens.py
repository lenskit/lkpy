# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Code to import MovieLens data sets into LensKit.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from io import TextIOWrapper
from pathlib import Path
from typing import TypeAlias
from zipfile import ZipFile

import numpy as np
import pandas as pd
import structlog
from scipy.sparse import coo_array

from lenskit.logging import get_logger

from .builder import DatasetBuilder
from .dataset import Dataset

_log = get_logger(__name__)

LOC: TypeAlias = Path | tuple[ZipFile, str]


class MLData:
    """
    Internal class representing an open ML data set.

    .. stability:: internal
    """

    version: str
    source: Path | ZipFile
    prefix: str = ""

    _logger: structlog.stdlib.BoundLogger

    def __init__(self, version: str, source: Path | ZipFile, prefix: str = ""):
        self.version = version
        self.source = source
        self.prefix = prefix

        log = _log.bind(version=version)
        if isinstance(source, ZipFile):
            log = log.bind(source=source.filename)
        else:
            log = log.bind(source=str(source))
        self._logger = log

    @staticmethod
    def version_impl(version: str) -> Callable[..., MLData]:
        if version == "ml-100k":
            return ML100KLoader
        elif re.match(r"^ml-10?m(100k)?$", version, re.IGNORECASE):
            return MLMLoader
        elif re.match(r"^ml-(\d+m|latest(-small)?)$", version, re.IGNORECASE):
            return MLModernLoader
        else:  # pragma: nocover
            raise ValueError(f"unknown ML version {version}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if isinstance(self.source, ZipFile):
            self.source.close()

    def open_file(self, name: str, encoding: str = "utf8"):
        if isinstance(self.source, Path):
            return open(self.source / (self.prefix + name), "rb")
        else:
            self._logger.debug("opening zip archive member")
            return self.source.open(self.prefix + name)

    def dataset(self) -> Dataset:  # pragma: nocover
        """
        Load the full dataset.
        """
        raise NotImplementedError()

    def ratings_df(self) -> pd.DataFrame:  # pragma: nocover
        """
        Load the ratings data frame.
        """
        raise NotImplementedError()


class ML100KLoader(MLData):
    """
    Loader for the ML100K data set.
    """

    def dataset(self) -> Dataset:
        dsb = DatasetBuilder(self.version)

        genres = self.genres().tolist()
        movies = self.movies_df(genres)
        movies = movies.drop(columns=["misc"])

        dsb.add_entities("item", movies["item_id"])
        dsb.add_scalar_attribute("item", "title", movies["item_id"], movies["title"])
        # TODO: add movie genres

        ratings = self.ratings_df()
        dsb.add_interactions(
            "rating", ratings, entities=["user", "item"], missing="insert", default=True
        )

        users = self.users_df().set_index("user_id")
        dsb.add_entities("user", users.index.values, duplicates="update")
        for c in users.columns:
            self._logger.debug("adding user column %s", c)
            dsb.add_scalar_attribute("user", c, users[c])

        return dsb.build()

    def genres(self) -> pd.Series:
        self._logger.debug("reading ML100K genre list")
        with self.open_file("u.genre") as data:
            df = pd.read_csv(data, sep="|", names=["name", "number"])
            return df.set_index("number").sort_index()["name"]

    def movies_df(self, genres: list[str] | None = None) -> pd.DataFrame:
        if genres is None:
            genres = self.genres().tolist()
        self._logger.debug("reading ML100K movie info")
        with self.open_file("u.item") as data:
            return pd.read_csv(
                data,
                sep=r"\|",
                header=None,
                names=["item_id", "title", "date", "misc", "IMDB"] + genres,
                dtype={
                    "item_id": np.int32,
                },
                encoding="latin1",
            )

    def users_df(self) -> pd.DataFrame:
        self._logger.debug("reading ML100K user info")
        with self.open_file("u.user") as data:
            return pd.read_csv(
                data,
                sep=r"\|",
                header=None,
                names=["user_id", "age", "gender", "occupation", "zip_code"],
                dtype={
                    "user_id": np.int32,
                },
                encoding="latin1",
            )

    def ratings_df(self) -> pd.DataFrame:
        self._logger.debug("reading ML100K ratings TSV")
        with self.open_file("u.data") as data:
            df = pd.read_csv(
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
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df


class MLMLoader(MLData):
    """
    Loader for the ML 1M and 10M data sets.
    """

    def dataset(self) -> Dataset:
        dsb = DatasetBuilder(self.version)

        movies = self.movies_df()

        dsb.add_entities("item", movies["item_id"])
        dsb.add_scalar_attribute("item", "title", movies["item_id"], movies["title"])
        genres = movies["genres"].str.split("|", regex=False)
        dsb.add_list_attribute("item", "genres", movies["item_id"], genres, dictionary=True)

        ratings = self.ratings_df()
        dsb.add_interactions(
            "rating", ratings, entities=["user", "item"], missing="insert", default=True
        )

        users = self.users_df()
        if users is not None:
            users = users.set_index("user_id")
            dsb.add_entities("user", users.index.values, duplicates="update")
            for c in users.columns:
                self._logger.debug("adding user column %s", c)
                dsb.add_scalar_attribute("user", c, users[c])

        tags = self.tagging_df()
        if tags is not None:
            # we add them as a sparse vector attribute of counts
            tag_names = np.unique(tags["tag"])
            tag_idx = pd.Index(tag_names)
            tags["tag_num"] = tag_idx.get_indexer_for(tags["tag"])
            tag_counts = (
                tags.groupby(["item_id", "tag_num"])["tag"].count().reset_index(name="count")
            )
            tag_items = np.unique(tags["item_id"])
            item_idx = pd.Index(tag_items)
            icol = item_idx.get_indexer_for(tag_counts["item_id"])
            tag_matrix = coo_array((tag_counts["count"], (icol, tag_counts["tag_num"])))
            dsb.add_vector_attribute("item", "tag_counts", tag_items, tag_matrix)

        return dsb.build()

    def movies_df(self):
        self._logger.debug("reading ML10?M movies file")
        with self.open_file("movies.dat") as data:
            return pd.read_csv(
                data,
                sep=":",
                header=None,
                names=["item_id", "_it", "title", "_tg", "genres"],
                usecols=[0, 2, 4],
                dtype={
                    "item_id": np.int32,
                },
                encoding="latin1",
            )

    def users_df(self):
        if self.version != "ml-1m":
            return None

        self._logger.debug("reading ML1M users file")
        with self.open_file("users.dat") as data:
            return pd.read_csv(
                data,
                sep=":",
                header=None,
                names=[
                    "user_id",
                    "_ug",
                    "gender",
                    "_ga",
                    "age",
                    "_ao",
                    "occupation",
                    "_az",
                    "zip_code",
                ],
                usecols=[0, 2, 4, 6, 8],
                dtype={
                    "user_id": np.int32,
                },
                encoding="latin1",
            )

    def ratings_df(self):
        self._logger.debug("reading ML10?M ratings file")
        with self.open_file("ratings.dat") as data:
            df = pd.read_csv(
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
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df

    def tagging_df(self):
        if self.version != "ml-10m":
            return None

        self._logger.debug("reading ML10M tags file")
        # this is slow but the data is a mess
        lpat = re.compile(r"^(\d+)::(\d+)::(.*)::(\d+)$")
        lines = []
        with self.open_file("tags.dat") as data:
            for i, line in enumerate(TextIOWrapper(data, "latin1"), 1):
                m = lpat.match(line)
                if not m:  # pragma: nocover
                    self._logger.warn("invalid line", line=i)
                    continue

                lines.append((int(m[1]), int(m[2]), m[3], int(m[4])))

        df = pd.DataFrame.from_records(lines, columns=["user_id", "item_id", "tag", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df


class MLModernLoader(MLData):
    """
    Loader for modern MovieLens data sets (20M and later).
    """

    def dataset(self) -> Dataset:
        dsb = DatasetBuilder(self.version)

        movies = self.movies_df()

        dsb.add_entities("item", movies["item_id"])
        dsb.add_scalar_attribute("item", "title", movies["item_id"], movies["title"])
        genres = movies["genres"].str.split("|", regex=False)
        dsb.add_list_attribute("item", "genres", movies["item_id"], genres)

        ratings = self.ratings_df()
        dsb.add_interactions(
            "rating", ratings, entities=["user", "item"], missing="insert", default=True
        )

        tags = self.tagging_df()
        # we add them as a sparse vector attribute of counts
        tag_names = np.unique(tags["tag"])
        tag_idx = pd.Index(tag_names)
        tags["tag_num"] = tag_idx.get_indexer_for(tags["tag"])
        tag_counts = tags.groupby(["item_id", "tag_num"])["tag"].count().reset_index(name="count")
        tag_items = np.unique(tags["item_id"])
        item_idx = pd.Index(tag_items)
        icol = item_idx.get_indexer_for(tag_counts["item_id"])
        tag_matrix = coo_array((tag_counts["count"], (icol, tag_counts["tag_num"])))
        dsb.add_vector_attribute("item", "tag_counts", tag_items, tag_matrix)

        genome = self.genome_df()
        if genome is not None:
            dsb.add_vector_attribute(
                "item",
                "tag_genome",
                genome.index.values,
                genome.to_numpy(),
                dim_names=genome.columns,
            )

        return dsb.build()

    def movies_df(self):
        self._logger.debug("reading modern movies CSV")
        with self.open_file("movies.csv") as data:
            df = pd.read_csv(
                data,
                dtype={
                    "movieId": np.int32,
                },
            ).rename(columns={"movieId": "item_id"})
            return df

    def tagging_df(self):
        self._logger.debug("reading modern movies CSV")
        with self.open_file("tags.csv") as data:
            df = pd.read_csv(
                data,
                dtype={"userId": np.int32, "movieId": np.int32, "tag": "string"},
            ).rename(columns={"userId": "user_id", "movieId": "item_id"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df[df["tag"].notnull()]
            return df

    def genome_df(self):
        if isinstance(self.source, Path):
            if not (self.source / "genome-tags.csv").exists():
                return None
        else:
            name = self.prefix + "genome-tags.csv"
            if name not in self.source.namelist():
                return None

        with self.open_file("genome-tags.csv") as data:
            tags = pd.read_csv(
                data,
                dtype={"tagId": np.int32, "tag": "string"},
            )
        with self.open_file("genome-scores.csv") as data:
            scores = pd.read_csv(
                data, dtype={"movieId": np.int32, "tagId": np.int32, "relevance": np.float32}
            ).rename(columns={"movieId": "item_id"})

        tag_names = tags.set_index("tagId")["tag"].to_dict()
        df = pd.pivot(scores, columns="tagId", index="item_id", values="relevance")
        df = df.rename(columns=tag_names)
        return df

    def ratings_df(self):
        self._logger.debug("reading modern ratings CSV")
        with self.open_file("ratings.csv") as data:
            df = pd.read_csv(
                data,
                dtype={
                    "userId": np.int32,
                    "movieId": np.int32,
                    "rating": np.float32,
                    "timestamp": np.int64,
                },
            ).rename(columns={"userId": "user_id", "movieId": "item_id"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df


def load_movielens(path: str | Path) -> Dataset:
    """
    Load a MovieLens dataset.  The appropriate MovieLens format is detected
    based on the file contents.

    Stability:
        Caller

    Args:
        path:
            The path to the dataset, either as an unpacked directory or a zip
            file.

    Returns:
        The dataset.
    """
    with _ml_detect_and_open(path) as ml:
        return ml.dataset()


def load_movielens_df(path: str | Path) -> pd.DataFrame:
    """
    Load the ratings from a MovieLens dataset as a raw data frame.  The
    appropriate MovieLens format is detected based on the file contents.

    Stability:
        Caller

    Args:
        path:
            The path to the dataset, either as an unpacked directory or a zip
            file.

    Returns:
        The ratings, with columns ``user_id``, ``item_id``, ``rating``, and
        ``timestamp``.
    """
    with _ml_detect_and_open(path) as ml:
        return ml.ratings_df()


def _ml_detect_and_open(path: str | Path) -> MLData:
    loc = Path(path)
    version: str
    ctor: Callable[..., MLData]

    if loc.is_file() and loc.suffix == ".zip":
        log = _log.bind(zipfile=str(loc))
        log.debug("opening zip file")
        zf = ZipFile(loc, "r")
        try:
            infos = zf.infolist()
            first = infos[0]
            if not first.is_dir:  # pragma: nocover
                log.error("first entry is not directory")
                raise RuntimeError("invalid ML zip file")

            log.debug("base dir filename %s", first.filename)
            dsm = re.match(r"^(ml-(?:\d+[MmKk]|latest|latest-small))", first.filename)
            if not dsm:  # pragma: nocover
                log.error("invalid directory name %s", first.filename)
                raise RuntimeError("invalid ML zip file")

            version = dsm.group(1).lower()
            log.debug("found ML data set %s", version)
            ctor = MLData.version_impl(version)
            return ctor(version, zf, first.filename)
        except Exception as e:  # pragma nocover
            zf.close()
            raise e
    elif loc.is_dir():
        log = _log.bind(dir=str(loc))
        log.debug("loading from directory")
        dsm = re.match(r"^(ml-\d+[MmKk])", loc.name)
        if dsm:
            version = dsm.group(1)
            ctor = MLData.version_impl(dsm.group(1).lower())
            log.debug("inferred data set %s from dir name", version)
        else:
            log.debug("checking contents for data type")
            if (loc / "u.data").exists():
                log.debug("found u.data, interpreting as 100K")
                ctor = ML100KLoader
            elif (loc / "ratings.dat").exists():
                if (loc / "tags.dat").exists():
                    log.debug("found ratings.dat and tags.dat, interpreting as 10M")
                    version = "ml-10m"
                else:
                    log.debug("found ratings.dat but no tags, interpreting as 1M")
                    version = "ml-1m"
                ctor = MLMLoader
            elif (loc / "ratings.csv").exists():
                log.debug("found ratings.csv, interpreting as modern (20M and later)")
                version = "ml-modern"
                ctor = MLModernLoader
            else:
                log.error("could not detect MovieLens data")
                raise RuntimeError("invalid ML directory")

        return ctor(version, loc)

    elif loc.exists():  # pragma: nocover
        _log.error("invalid MovieLens data location", path=path)
        raise RuntimeError("not a directory or zip file")
    else:  # pragma: nocover
        _log.error("MovieLens data not found", path=path)
        raise FileNotFoundError(path)
