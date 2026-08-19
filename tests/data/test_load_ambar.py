# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pandas as pd

from pytest import mark

from lenskit.data.sources.ambar import load_ambar

AMBAR_DIR = Path("data/ambar")
RATINGS_FILE = AMBAR_DIR / "ratings_info.csv"
USERS_FILE = AMBAR_DIR / "users_info.csv"
TRACKS_FILE = AMBAR_DIR / "tracks_info.csv"
ARTISTS_FILE = AMBAR_DIR / "artists_info.csv"


@mark.skipif(not AMBAR_DIR.exists(), reason="input data does not exist")
@mark.realdata
def test_ambar():
    ratings = pd.read_csv(RATINGS_FILE)
    users = pd.read_csv(USERS_FILE)
    tracks = pd.read_csv(TRACKS_FILE)
    artists = pd.read_csv(ARTISTS_FILE)

    data = load_ambar(AMBAR_DIR)

    assert data.interaction_count == len(ratings)
    assert data.user_count == users["user_id"].nunique()
    assert data.item_count == tracks["track_id"].nunique()
    assert data.entities("artist").count() == artists["artist_id"].nunique()


@mark.skipif(not AMBAR_DIR.exists(), reason="input data does not exist")
@mark.realdata
def test_ambar_artist_attribute():
    tracks = pd.read_csv(TRACKS_FILE)
    data = load_ambar(AMBAR_DIR)

    items = data.entities("item")
    assert "artist_id" in items.attributes

    artist_attr = items.attribute("artist_id")
    assert len(artist_attr) == data.item_count
    assert artist_attr.arrow().null_count == 0
