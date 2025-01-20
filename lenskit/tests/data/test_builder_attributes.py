# pyright: basic
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.sparse import csr_array

from pytest import approx, raises, warns

from lenskit.data import DatasetBuilder
from lenskit.diagnostics import DataError, DataWarning
from lenskit.testing import ml_test_dir

FRUITS = ["apple", "banana", "orange", "lemon", "mango"]


def test_item_scalar_series():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"}).set_index("item_id")

    dsb.add_entities("item", items.index.values)
    dsb.add_scalar_attribute("item", "title", items["title"])
    sa = dsb.schema.entities["item"].attributes["title"]
    assert sa.layout == "scalar"

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
    sa = dsb.schema.entities["item"].attributes["title"]
    assert sa.layout == "scalar"

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
    sa = dsb.schema.entities["item"].attributes["title"]
    assert sa.layout == "scalar"

    ds = dsb.build()

    df = ds.entities("item").attribute("title").series()
    assert isinstance(df, pd.Series)
    assert df == items["title"]


def test_item_insert_with_scalar_df():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})

    dsb.add_entities("item", items.index.values)
    dsb.add_scalar_attribute("item", "title", items["item_id"], items["title"])
    sa = dsb.schema.entities["item"].attributes["title"]
    assert sa.layout == "scalar"

    ds = dsb.build()

    df = ds.entities("item").attribute("title").series()
    assert isinstance(df, pd.Series)
    assert df == items["title"]


def test_item_list_series():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"}).set_index("item_id")

    genres = items["genres"].str.split("|")

    dsb.add_entities("item", items.index.values)
    dsb.add_list_attribute("item", "genres", genres)
    va = dsb.schema.entities["item"].attributes["genres"]
    assert va.layout == "list"

    ds = dsb.build()

    arr = ds.entities("item").attribute("genres").arrow()
    assert isinstance(arr, pa.ListArray)
    assert len(arr) == len(genres)
    assert np.all(arr.is_valid())

    gs = ds.entities("item").attribute("genres").series()
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
    va = dsb.schema.entities["item"].attributes["genres"]
    assert va.layout == "list"

    ds = dsb.build()

    arr = ds.entities("item").attribute("genres").arrow()
    assert isinstance(arr, pa.ListArray)
    assert len(arr) == len(items)


def test_item_initial_list_df():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})
    items["genres"] = items["genres"].str.split("|")

    dsb.add_entities("item", items)
    va = dsb.schema.entities["item"].attributes["genres"]
    assert va.layout == "list"

    ds = dsb.build()

    arr = ds.entities("item").attribute("title").arrow()
    assert isinstance(arr, pa.StringArray)
    assert len(arr)

    arr = ds.entities("item").attribute("genres").arrow()
    assert isinstance(arr, pa.ListArray)
    assert len(arr) == len(items)


def test_item_vector(rng: np.random.Generator, ml_df: pd.DataFrame):
    dsb = DatasetBuilder()

    item_ids = ml_df["item"].unique()
    dsb.add_entities("item", item_ids)

    items = rng.choice(item_ids, 500, replace=False)
    vec = rng.standard_normal((500, 20))
    dsb.add_vector_attribute("item", "embedding", items, vec)
    va = dsb.schema.entities["item"].attributes["embedding"]
    assert va.layout == "vector"

    ds = dsb.build()
    assert ds.entities("item").attribute("embedding").names is None

    arr = ds.entities("item").attribute("embedding").arrow()
    assert isinstance(arr, pa.FixedSizeListArray)
    assert np.sum(arr.is_valid()) == 500
    assert np.all(np.asarray(arr.value_lengths()) == 20)

    arr = ds.entities("item").attribute("embedding").numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (len(item_ids), 20)
    assert np.sum(np.isfinite(arr)) == 500 * 20

    arr = ds.entities("item").attribute("embedding").df(missing="omit")
    assert isinstance(arr, pd.DataFrame)
    assert np.all(arr.columns == np.arange(20))
    assert arr.shape == (500, 20)
    assert set(arr.index) == set(items)


def test_item_vector_names(rng: np.random.Generator, ml_df: pd.DataFrame):
    dsb = DatasetBuilder()

    item_ids = ml_df["item"].unique()
    dsb.add_entities("item", item_ids)

    items = rng.choice(item_ids, 500, replace=False)
    vec = rng.standard_normal((500, 5))
    dsb.add_vector_attribute("item", "embedding", items, vec, dims=FRUITS)
    va = dsb.schema.entities["item"].attributes["embedding"]
    assert va.layout == "vector"

    ds = dsb.build()
    assert ds.entities("item").attribute("embedding").names == FRUITS

    arr = ds.entities("item").attribute("embedding").arrow()
    assert isinstance(arr, pa.FixedSizeListArray)
    assert np.sum(arr.is_valid()) == 500
    assert np.all(np.asarray(arr.value_lengths()) == 5)

    arr = ds.entities("item").attribute("embedding").numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (len(item_ids), 20)
    assert np.sum(np.isfinite(arr)) == 500 * 20

    arr = ds.entities("item").attribute("embedding").df(missing="omit")
    assert isinstance(arr, pd.DataFrame)
    assert np.all(arr.columns == FRUITS)
    assert arr.shape == (500, 20)
    assert set(arr.index) == set(items)


def test_item_sparse_attribute(rng: np.random.Generator, ml_df: pd.DataFrame):
    dsb = DatasetBuilder()

    movies = pd.read_csv(ml_test_dir / "movies.csv")
    movies = movies.rename(columns={"movieId": "item_id"}).set_index("item_id")

    dsb.add_entities("item", movies.index)

    genres = movies["genres"].str.split("|").reset_index().explode("genres", ignore_index=True)
    gindex = pd.Index(np.unique(genres["genres"]))

    ig_rows = movies.index.get_indexer_for(genres["item_id"])
    ig_cols = gindex.get_indexer_for(genres["genres"])
    ig_vals = np.ones(len(ig_rows), np.int32)

    arr = csr_array((ig_vals, (ig_rows, ig_cols)))
    dsb.add_vector_attribute("item", "genres", movies.index, arr)

    ga = dsb.schema.entities["item"].attributes["genres"]
    assert ga.layout == "sparse"

    items = rng.choice(movies.index, 500, replace=False)
    vec = rng.standard_normal((500, 5))
    dsb.add_vector_attribute("item", "embedding", items, vec, dims=FRUITS)

    ds = dsb.build()
    assert ds.entities("item").attribute("embedding").names == FRUITS

    arr = ds.entities("item").attribute("embedding").arrow()
    assert isinstance(arr, pa.FixedSizeListArray)
    assert np.sum(arr.is_valid()) == 500
    assert np.all(np.asarray(arr.value_lengths()) == 5)

    arr = ds.entities("item").attribute("embedding").numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (len(movies.index), 20)
    assert np.sum(np.isfinite(arr)) == 500 * 20

    arr = ds.entities("item").attribute("embedding").df(missing="omit")
    assert isinstance(arr, pd.DataFrame)
    assert np.all(arr.columns == FRUITS)
    assert arr.shape == (500, 20)
    assert set(arr.index) == set(items)
