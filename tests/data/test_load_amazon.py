# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from pytest import mark

from lenskit.data import load_amazon_ratings
from lenskit.data.vocab import Vocabulary

AZ_NAME = os.environ.get("LK_TEST_AZ_SET", "Video_Games")

AZ14_TEST = Path(f"data/az14/ratings_{AZ_NAME}.csv")
AZ18_TEST = Path(f"data/az18/{AZ_NAME}.csv")
AZ23_TEST = Path(f"data/az23/{AZ_NAME}.csv.gz")


@mark.skipif(not AZ14_TEST.exists(), reason="input data does not exist")
@mark.xfail(reason="duplicate ratings")
@mark.realdata
def test_amazon_2014():
    df = pd.read_csv(AZ14_TEST, names=["user_id", "item_id", "rating", "timestamp"])
    data = load_amazon_ratings(AZ14_TEST)

    assert data.interaction_count == len(df)
    assert data.user_count == df["user_id"].nunique()
    assert data.item_count == df["item_id"].nunique()


@mark.skipif(not AZ18_TEST.exists(), reason="input data does not exist")
@mark.xfail(reason="duplicate ratings")
@mark.realdata
def test_amazon_2018():
    df = pd.read_csv(AZ18_TEST, names=["item_id", "user_id", "rating", "timestamp"])
    data = load_amazon_ratings(AZ18_TEST)

    assert data.interaction_count == len(df)
    assert data.user_count == df["user_id"].nunique()
    assert data.item_count == df["item_id"].nunique()


@mark.skipif(not AZ23_TEST.exists(), reason="input data does not exist")
@mark.realdata
def test_amazon_2023_vocab():
    "targeted vocab tests for broken data set"
    df = pd.read_csv(AZ23_TEST)

    user_ids = df["user_id"].unique()
    item_ids = df["parent_asin"].unique()

    user_vocab = Vocabulary(user_ids)
    unos = user_vocab.numbers(df["user_id"], format="arrow", missing="null")
    assert unos.null_count == 0
    assert pc.count_distinct(unos).as_py() == len(user_ids)

    item_vocab = Vocabulary(item_ids)
    inos = item_vocab.numbers(df["parent_asin"], format="arrow", missing="null")
    assert inos.null_count == 0
    assert pc.count_distinct(inos).as_py() == len(item_ids)


@mark.skipif(not AZ23_TEST.exists(), reason="input data does not exist")
@mark.realdata
def test_amazon_2023():
    df = pd.read_csv(AZ23_TEST)
    data = load_amazon_ratings(AZ23_TEST)

    assert data.interaction_count == len(df)
    assert data.user_count == df["user_id"].nunique()
    assert data.item_count == df["parent_asin"].nunique()
