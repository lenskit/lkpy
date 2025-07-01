# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: basic
import numpy as np
import pandas as pd
import pyarrow as pa

from pytest import approx, mark, raises, skip, warns

from lenskit.data import Dataset, DatasetBuilder


def test_binarize_zero(ml_ds: Dataset) -> None:
    """Test that binarize removes items."""
    ratings_df = ml_ds.interactions().pandas()
    dsb = DatasetBuilder()
    ratings_df["user"] = ratings_df["user"].astype(int)
    ratings_df["item"] = ratings_df["item"].astype(int)
    dsb.add_interactions(
        "rating", ratings_df, entities=["user", "item"], missing="insert", default=True
    )

    dsb.binarize_ratings("rating", min_pos_rating=3.5, method="zero")

    ds = dsb.build()
    ratings = ds.interactions().pandas()
    assert set(ratings["rating"]) <= {0.0, 1.0}
    assert all((ratings["rating"] == 1.0) == (ratings_df["rating"] >= 3.5))


def test_binarize_remove(ml_ds: Dataset) -> None:
    ratings_df = ml_ds.interactions().pandas()
    dsb = DatasetBuilder()
    ratings_df["user"] = ratings_df["user"].astype(int)
    ratings_df["item"] = ratings_df["item"].astype(int)
    dsb.add_interactions(
        "rating", ratings_df, entities=["user", "item"], missing="insert", default=True
    )

    dsb.binarize_ratings("rating", min_pos_rating=3.5, method="remove")

    ds = dsb.build()
    ratings = ds.interactions().pandas()
    assert all(ratings["rating"] >= 3.5)


def test_binarize_error(ml_ds: Dataset) -> None:
    ratings_df = ml_ds.interactions().pandas()
    dsb = DatasetBuilder()
    ratings_df["user"] = ratings_df["user"].astype(int)
    ratings_df["item"] = ratings_df["item"].astype(int)
    dsb.add_interactions(
        "rating", ratings_df, entities=["user", "item"], missing="insert", default=True
    )

    with raises(ValueError, match="method must be 'zero' or 'remove'"):
        dsb.binarize_ratings("rating", min_pos_rating=3.5, method="unknown")

    with raises(ValueError, match="min_pos_rating 6.0 is outside the rating range"):
        dsb.binarize_ratings("rating", min_pos_rating=6.0, method="zero")

    with raises(ValueError, match="min_pos_rating -1.0 is outside the rating range"):
        dsb.binarize_ratings("rating", min_pos_rating=-1.0, method="zero")
