# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
from numpy.random import Generator

from lenskit.data import Dataset


def test_all_entities(rng: Generator, ml_ratings: pd.DataFrame, ml_ds: Dataset):
    assert len(ml_ds.entities("item")) == ml_ds.item_count
    assert len(ml_ds.entities("item")) >= ml_ratings["item_id"].nunique()
    assert len(ml_ds.entities("user")) == ml_ratings["user_id"].nunique()

    stats = ml_ds.item_stats()

    assert np.all(
        ml_ds.entities("item").ids()[stats["count"] > 0] == np.unique(ml_ratings["item_id"])
    )
    assert np.all(ml_ds.entities("item").numbers() == np.arange(ml_ds.item_count))

    df = ml_ds.entities("item").pandas()
    assert len(df) >= ml_ratings["item_id"].nunique()
    assert np.all(df["item_id"] == ml_ds.items.ids())


def test_entity_subset_ids(rng: Generator, ml_ratings: pd.DataFrame, ml_ds: Dataset):
    item_ids = rng.choice(ml_ratings["item_id"].unique(), 20, replace=False)

    ents = ml_ds.entities("item").select(ids=item_ids)
    assert len(ents) == len(item_ids)
    assert np.all(ents.ids() == item_ids)
    assert np.all(ents.numbers() == ml_ds.items.numbers(item_ids))

    df = ents.pandas()
    assert np.all(df["item_id"] == item_ids)


def test_entity_subset_numbers(rng: Generator, ml_ratings: pd.DataFrame, ml_ds: Dataset):
    inos = rng.choice(ml_ratings["item_id"].nunique(), 20, replace=False)

    ents = ml_ds.entities("item").select(numbers=inos)
    assert len(ents) == len(inos)
    assert np.all(ents.numbers() == inos)
    assert np.all(ents.ids() == ml_ds.items.ids(inos))


def test_entity_subset_subset_numbers(rng: Generator, ml_ratings: pd.DataFrame, ml_ds: Dataset):
    "Test that subsetting a subset works."
    inos = rng.choice(ml_ratings["item_id"].nunique(), 100, replace=False)

    ents = ml_ds.entities("item").select(numbers=inos)
    assert len(ents) == len(inos)

    iss2 = rng.choice(inos, 20, replace=False)
    e2 = ents.select(numbers=iss2)
    assert len(e2) == 20
    assert np.all(e2.ids() == ml_ds.items.ids(iss2))
    assert np.all(e2.numbers() == iss2)


def test_entity_drop_null_attributes(rng: Generator, ml_ds: Dataset):
    attr = ml_ds.entities("item").attribute("tag_counts")
    a2 = attr.drop_null()

    assert len(a2) < len(attr)

    mat = a2.scipy()
    assert np.all(np.diff(mat.indptr) >= 1)
    assert mat.data.sum() > 1000
