# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Support for the AMBAR music recommendation dataset.
"""

from os import PathLike
from pathlib import Path

import pyarrow as pa
from pyarrow.csv import ConvertOptions, open_csv
from xopen import xopen

from lenskit.logging import get_logger

from .._builder import DatasetBuilder
from .._dataset import Dataset

_log = get_logger(__name__)


def _read_table(path: Path, **convert_kwargs) -> pa.Table:
    with (
        xopen(path, "rb") as f,
        open_csv(f, convert_options=ConvertOptions(**convert_kwargs)) as reader,
    ):
        return pa.Table.from_batches(reader, schema=reader.schema)


def _clean_user_duplicates(users: pa.Table) -> pa.Table:
    df = users.to_pandas()

    # exact duplicate rows collapse to one
    df = df.drop_duplicates()

    # still duplicated on user_id has a conflict
    conflict_mask = df["user_id"].duplicated(keep=False)
    n_conflicts = df.loc[conflict_mask, "user_id"].nunique()
    if n_conflicts:
        _log.warning("dropping %d users with conflicting duplicate records", n_conflicts)
        df = df[~conflict_mask]

    return pa.Table.from_pandas(df, preserve_index=False)


def load_ambar(path: Path | str | PathLike[str]) -> Dataset:
    """
    Load the AMBAR music dataset.

    AMBAR consists of roughly 3.3M ratings from around 31,013 users for
    443,921 tracks and 30,667 artists.  Ratings are on a 1-5 Likert scale,
    derived from quintiles of each user's listening-history playcounts.

    The dataset directory is expected to contain four files:
    ``users_info.csv``, ``tracks_info.csv``, ``artists_info.csv``, and
    ``ratings_info.csv``.

    Args:
        path:
            The directory containing the AMBAR data files.

    Returns:
        The loaded AMBAR dataset.
    """
    path = Path(path)
    log = _log.bind(path=str(path))

    dsb = DatasetBuilder("ambar")

    log.debug("reading user info")
    users = _read_table(path / "users_info.csv", column_types={"user_id": pa.int64()})
    users = _clean_user_duplicates(users)
    dsb.add_entities("user", users)

    log.debug("reading artist info")
    artists = _read_table(path / "artists_info.csv", column_types={"artist_id": pa.int64()})
    dsb.add_entities("artist", artists)

    log.debug("reading track info")
    tracks = _read_table(
        path / "tracks_info.csv",
        column_types={"track_id": pa.int64(), "artist_id": pa.int64()},
    )
    tracks = tracks.rename_columns(
        ["item_id" if c == "track_id" else c for c in tracks.column_names]
    )

    track_artist_ids = tracks.column("artist_id")
    track_ids = tracks.column("item_id")
    tracks_meta = tracks.drop_columns(["artist_id"])

    dsb.add_entities("item", tracks_meta)
    dsb.add_scalar_attribute("item", "artist_id", track_ids, track_artist_ids)

    log.debug("reading ratings")
    ratings = _read_table(
        path / "ratings_info.csv",
        column_types={"user_id": pa.int64(), "track_id": pa.int64(), "rating": pa.float32()},
    )
    ratings = ratings.rename_columns(["user_id", "item_id", "rating"])

    # user_ids that appear in ratings but have no users_info record at all
    valid_user_ids = set(users.column("user_id").to_pylist())
    ratings_df = ratings.to_pandas()
    ratings_df = ratings_df[ratings_df["user_id"].isin(valid_user_ids)]
    ratings = pa.Table.from_pandas(ratings_df, preserve_index=False)

    dsb.add_interactions(
        "rating", ratings, entities=["user", "item"], missing="error", allow_repeats=True
    )

    log.info(
        "loaded AMBAR data",
        n_users=users.num_rows,
        n_tracks=tracks.num_rows,
        n_artists=artists.num_rows,
        n_ratings=ratings.num_rows,
    )

    return dsb.build()
