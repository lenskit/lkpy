# pyright: basic
import numpy as np
import pandas as pd
import pyarrow as pa
from docutils import DataError

from pytest import approx, raises, warns

from lenskit.data import DatasetBuilder
from lenskit.diagnostics import DataError, DataWarning
from lenskit.testing import ml_test_dir


def test_item_scalar_series():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"}).set_index("item_id")

    dsb.add_entities("item", items.index.values)
    dsb.add_scalar_attribute("item", "title", items["title"])

    ds = dsb.build()

    df = ds.entities("item").attribute("title")
    assert isinstance(df, pd.Series)
    assert df == items["title"]


def test_item_scalar_df():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})

    dsb.add_entities("item", items.index.values)
    dsb.add_scalar_attribute("item", "title", items[["item_id", "title"]])

    ds = dsb.build()

    df = ds.entities("item").attribute("title")
    assert isinstance(df, pd.Series)
    assert df == items["title"]


def test_item_scalar_array():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})

    dsb.add_entities("item", items.index.values)
    dsb.add_scalar_attribute("item", "title", items["item_id"], items["title"])

    ds = dsb.build()

    df = ds.entities("item").attribute("title")
    assert isinstance(df, pd.Series)
    assert df == items["title"]


def test_item_list_series():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"}).set_index("item_id")

    genres = items["genres"].str.split("|")

    dsb.add_entities("item", items.index.values)
    dsb.add_list_attribute("item", "genres", genres)

    ds = dsb.build()

    arr = ds.entities("item").attribute("genres", format="arrow")
    assert isinstance(arr, pa.ListArray)
    assert len(arr) == len(genres)
    assert np.all(arr.is_valid())

    gs = ds.entities("item").attribute("genres", format="pandas")
    assert np.all(gs.index == ds.items.ids())
    gs, gs2 = gs.align(genres, how="outer")
    assert np.all(gs.notnull())
    assert np.all(gs2.notnull())
    assert np.all(gs == gs2)


def test_item_list_df():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})
    items["genres"] = items["genres"].str.split("|")

    dsb.add_entities("item", items.index.values)
    dsb.add_list_attribute("item", "genres", items[["item_id", "genres"]])

    ds = dsb.build()

    arr = ds.entities("item").attribute("genres", format="arrow")
    assert isinstance(arr, pa.ListArray)
    assert len(arr) == len(items)


def test_item_initial_list_df():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})
    items["genres"] = items["genres"].str.split("|")

    dsb.add_entities("item", items)

    ds = dsb.build()

    arr = ds.entities("item").attribute("title", format="arrow")
    assert isinstance(arr, pa.StringArray)
    assert len(arr)

    arr = ds.entities("item").attribute("genres", format="arrow")
    assert isinstance(arr, pa.ListArray)
    assert len(arr) == len(items)


def test_item_vector(rng: np.random.Generator, ml_df: pd.DataFrame):
    dsb = DatasetBuilder()

    item_ids = ml_df["item"].unique()
    dsb.add_entities("item", item_ids)

    items = rng.choice(item_ids, 500, replace=False)
    vec = rng.standard_normal((500, 20))
    dsb.add_vector_attribute("item", "embedding", items, vec)

    ds = dsb.build()
    arr = ds.entities("item").attribute("embedding", format="arrow")
    assert isinstance(arr, pa.FixedSizeListArray)
    assert np.sum(arr.is_valid()) == 500
    assert np.all(np.asarray(arr.value_lengths()) == 20)

    arr = ds.entities("item").attribute("embedding", format="numpy")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (len(item_ids), 20)
    assert np.sum(np.isfinite(arr)) == 500 * 20

    arr = ds.entities("item").attribute("embedding", format="pandas", missing="omit")
    assert isinstance(arr, pd.DataFrame)
    assert np.all(arr.columns == np.arange(20))
    assert arr.shape == (500, 20)
    assert set(arr.index) == set(items)
