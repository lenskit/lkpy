# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import functools
from pathlib import Path

from pytest import mark

from lenskit.data.movielens import load_movielens, load_movielens_df

ML_LATEST_DIR = Path("data/ml-latest-small")

ML_100K_ZIP = Path("data/ml-100k.zip")
ML_1M_ZIP = Path("data/ml-1m.zip")
ML_10M_ZIP = Path("data/ml-10m.zip")
ML_20M_ZIP = Path("data/ml-20m.zip")
ML_25M_ZIP = Path("data/ml-25m.zip")
ML_32M_ZIP = Path("data/ml-32m.zip")


def test_latest_small_dir():
    ds = load_movielens(ML_LATEST_DIR)
    assert ds.item_count >= 100
    assert ds.user_count >= 100
    assert ds.user_count < 1000
    assert ds.interaction_count >= 100_000


@mark.skipif(not ML_100K_ZIP.exists(), reason="ml-100k does not exist")
def test_100k_zip():
    ds = load_movielens(ML_100K_ZIP)
    assert ds.item_count >= 100
    assert ds.user_count >= 100
    assert ds.user_count < 1000
    assert ds.interaction_count >= 100_000

    titles = ds.entities("item").attribute("title")
    title_s = titles.pandas()
    assert title_s.loc[1] == "Toy Story (1995)"

    genders = ds.entities("user").attribute("gender").pandas()
    # only binary gender is recorded
    assert set(genders) == {"M", "F"}


@mark.skipif(not ML_100K_ZIP.exists(), reason="ml-100k does not exist")
def test_100k_df():
    ds = load_movielens_df(ML_100K_ZIP)
    assert ds["item_id"].nunique() >= 100
    assert ds["user_id"].nunique() >= 100
    assert ds["user_id"].nunique() < 1000
    assert len(ds) >= 100_000


@mark.skipif(not ML_1M_ZIP.exists(), reason="ml-1m does not exist")
def test_1m_zip():
    ds = load_movielens(ML_1M_ZIP)
    assert ds.item_count >= 500
    assert ds.user_count >= 500
    assert ds.interaction_count >= 1_000_000

    titles = ds.entities("item").attribute("title")
    title_s = titles.pandas()
    assert title_s.loc[1] == "Toy Story (1995)"

    genres = ds.entities("item").attribute("genres").pandas()
    # Cry, The Beloved Country is drama
    assert genres.loc[40] == ["Drama"]

    genders = ds.entities("user").attribute("gender").pandas()
    # only binary gender is recorded
    assert set(genders) == {"M", "F"}


@mark.skipif(not ML_1M_ZIP.exists(), reason="ml-1m does not exist")
def test_1m_df():
    ds = load_movielens_df(ML_1M_ZIP)
    assert ds["item_id"].nunique() >= 500
    assert ds["user_id"].nunique() >= 500
    assert len(ds) >= 1_000_000


@mark.skipif(not ML_10M_ZIP.exists(), reason="ml-10m does not exist")
def test_10m_zip():
    ds = load_movielens(ML_10M_ZIP)
    assert ds.item_count >= 10_000
    assert ds.user_count >= 69_000
    assert ds.interaction_count >= 10_000_000

    titles = ds.entities("item").attribute("title")
    title_s = titles.pandas()
    assert title_s.loc[1] == "Toy Story (1995)"

    genres = ds.entities("item").attribute("genres").pandas()
    # Cry, The Beloved Country is drama
    assert genres.loc[40] == ["Drama"]


@mark.skipif(not ML_10M_ZIP.exists(), reason="ml-10m does not exist")
def test_10m_df():
    ds = load_movielens_df(ML_10M_ZIP)
    assert ds["item_id"].nunique() >= 10_000
    assert ds["user_id"].nunique() >= 69_000
    assert len(ds) >= 10_000_000


@mark.skipif(not ML_20M_ZIP.exists(), reason="ml-20m does not exist")
def test_20m_zip():
    ds = load_movielens(ML_20M_ZIP)
    assert ds.item_count >= 25_000
    assert ds.user_count >= 130_000
    assert ds.interaction_count >= 20_000_000

    titles = ds.entities("item").attribute("title")
    title_s = titles.pandas()
    assert title_s.loc[1] == "Toy Story (1995)"


@mark.skipif(not ML_20M_ZIP.exists(), reason="ml-20m does not exist")
def test_20m_df():
    ds = load_movielens_df(ML_20M_ZIP)
    assert ds["item_id"].nunique() >= 25_000
    assert ds["user_id"].nunique() >= 130_000
    assert len(ds) >= 20_000_000


@mark.skipif(not ML_25M_ZIP.exists(), reason="ml-25m does not exist")
def test_25m_zip():
    ds = load_movielens(ML_25M_ZIP)
    assert ds.item_count >= 50_000
    assert ds.user_count >= 160_000
    assert ds.interaction_count >= 25_000_000

    titles = ds.entities("item").attribute("title")
    title_s = titles.pandas()
    assert title_s.loc[1] == "Toy Story (1995)"


@mark.skipif(not ML_25M_ZIP.exists(), reason="ml-25m does not exist")
def test_25m_df():
    ds = load_movielens_df(ML_25M_ZIP)
    assert ds["item_id"].nunique() >= 50_000
    assert ds["user_id"].nunique() >= 160_000
    assert len(ds) >= 25_000_000


@mark.skipif(not ML_32M_ZIP.exists(), reason="ml-32m does not exist")
def test_32m_zip():
    ds = load_movielens(ML_32M_ZIP)
    assert ds.item_count >= 50_000
    assert ds.user_count >= 200_000
    assert ds.interaction_count >= 32_000_000

    titles = ds.entities("item").attribute("title")
    title_s = titles.pandas()
    assert title_s.loc[1] == "Toy Story (1995)"


@mark.skidf(not ML_32M_ZIP.exists(), reason="ml-32m does not exist")
def test_32m_zip_df():
    ds = load_movielens_df(ML_32M_ZIP)
    assert ds["item_id"].nunique() >= 50_000
    assert ds["user_id"].nunique() >= 200_000
    assert len(ds) >= 32_000_000
