# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import functools
from pathlib import Path

from pytest import mark

from lenskit.data.movielens import load_movielens

ML_LATEST_DIR = Path("data/ml-latest-small")
ML_100K_DIR = Path("data/ml-100k")
ML_1M_DIR = Path("data/ml-1m")
ML_10M_DIR = Path("data/ml-10m")
ML_20M_DIR = Path("data/ml-20m")

ML_100K_ZIP = Path("data/ml-100k.zip")
ML_1M_ZIP = Path("data/ml-1m.zip")
ML_10M_ZIP = Path("data/ml-10m.zip")
ML_20M_ZIP = Path("data/ml-20m.zip")


@mark.skipif(not ML_LATEST_DIR.exists(), reason="ml-latest-small does not exist")
def test_latest_small_dir():
    ds = load_movielens(ML_LATEST_DIR)
    assert ds.item_count >= 100
    assert ds.user_count >= 100
    assert ds.user_count < 1000
    assert ds.interaction_count >= 100_000


@mark.skipif(not ML_100K_DIR.exists(), reason="ml-100k does not exist")
def test_100k_dir():
    ds = load_movielens(ML_100K_DIR)
    assert ds.item_count >= 100
    assert ds.user_count >= 100
    assert ds.user_count < 1000
    assert ds.interaction_count >= 100_000


@mark.skipif(not ML_1M_DIR.exists(), reason="ml-1m does not exist")
def test_1m_dir():
    ds = load_movielens(ML_1M_DIR)
    assert ds.item_count >= 500
    assert ds.user_count >= 500
    assert ds.interaction_count >= 1_000_000


@mark.skipif(not ML_10M_DIR.exists(), reason="ml-10m does not exist")
def test_10m_dir():
    ds = load_movielens(ML_10M_DIR)
    assert ds.item_count >= 100
    assert ds.user_count >= 100
    assert ds.interaction_count >= 10_000_000


@mark.skipif(not ML_20M_DIR.exists(), reason="ml-20m does not exist")
def test_20m_dir():
    ds = load_movielens(ML_20M_DIR)
    assert ds.item_count >= 100
    assert ds.user_count >= 100
    assert ds.interaction_count >= 20_000_000


@mark.skipif(not ML_100K_ZIP.exists(), reason="ml-100k does not exist")
def test_100k_zip():
    ds = load_movielens(ML_100K_ZIP)
    assert ds.item_count >= 100
    assert ds.user_count >= 100
    assert ds.user_count < 1000
    assert ds.interaction_count >= 100_000


@mark.skipif(not ML_1M_ZIP.exists(), reason="ml-1m does not exist")
def test_1m_zip():
    ds = load_movielens(ML_1M_ZIP)
    assert ds.item_count >= 500
    assert ds.user_count >= 500
    assert ds.interaction_count >= 1_000_000


@mark.skipif(not ML_10M_ZIP.exists(), reason="ml-10m does not exist")
def test_10m_zip():
    ds = load_movielens(ML_10M_ZIP)
    assert ds.item_count >= 100
    assert ds.user_count >= 100
    assert ds.interaction_count >= 10_000_000


@mark.skipif(not ML_20M_ZIP.exists(), reason="ml-20m does not exist")
def test_20m_zip():
    ds = load_movielens(ML_20M_ZIP)
    assert ds.item_count >= 100
    assert ds.user_count >= 100
    assert ds.interaction_count >= 20_000_000
