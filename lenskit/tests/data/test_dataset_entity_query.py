import numpy as np
import pandas as pd
from numpy.random import Generator

from lenskit.data import Dataset


def test_all_entities(rng: Generator, ml_ratings: pd.DataFrame, ml_ds: Dataset):
    assert len(ml_ds.entities("item")) == ml_ratings["item_id"].nunique()
    assert len(ml_ds.entities("user")) == ml_ratings["item_id"].nunique()

    assert np.all(ml_ds.entities("item").ids() == np.unique(ml_ratings["item_id"]))
    assert np.all(ml_ds.entities("item").numbers() == np.arange(ml_ds.item_count))


def test_entity_subset_ids(rng: Generator, ml_ratings: pd.DataFrame, ml_ds: Dataset):
    item_ids = rng.choice(ml_ratings["item_id"].unique(), 20, replace=False)

    ents = ml_ds.entities("item").select(ids=item_ids)
    assert len(ents) == len(item_ids)
    assert np.all(ents.ids() == item_ids)
    assert np.all(ents.numbers() == ml_ds.items.numbers(item_ids))


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
