# pyright: basic
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from scipy.sparse import csr_array

from pytest import approx, mark, raises, skip, warns

from lenskit.data import DatasetBuilder
from lenskit.data.schema import AttrLayout
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
    assert sa.layout == AttrLayout.SCALAR

    ds = dsb.build()

    df = ds.entities("item").pandas().set_index("item_id")
    assert "title" in df.columns
    ds_ts, orig_ts = df["title"].align(items["title"])
    assert np.all(ds_ts == orig_ts)

    assert ds.entities("item").attribute("title").is_scalar
    df = ds.entities("item").attribute("title").pandas()
    assert isinstance(df, pd.Series)
    assert np.all(df == items["title"])


def test_item_scalar_df():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})

    dsb.add_entities("item", items["item_id"].values)
    dsb.add_scalar_attribute("item", "title", items[["item_id", "title"]])
    sa = dsb.schema.entities["item"].attributes["title"]
    assert sa.layout == AttrLayout.SCALAR

    ds = dsb.build()

    df = ds.entities("item").pandas().set_index("item_id")
    assert "title" in df.columns
    ds_ts, orig_ts = df["title"].align(items.set_index("item_id")["title"])
    assert np.all(ds_ts == orig_ts)

    assert ds.entities("item").attribute("title").is_scalar
    df = ds.entities("item").attribute("title").pandas()
    assert isinstance(df, pd.Series)
    ds_ts, orig_ts = df.align(items.set_index("item_id")["title"])
    assert np.all(ds_ts == orig_ts)


def test_item_scalar_array():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})

    dsb.add_entities("item", items["item_id"].values)
    dsb.add_scalar_attribute("item", "title", items["item_id"], items["title"])
    sa = dsb.schema.entities["item"].attributes["title"]
    assert sa.layout == AttrLayout.SCALAR

    ds = dsb.build()

    df = ds.entities("item").pandas().set_index("item_id")
    assert "title" in df.columns
    ds_ts, orig_ts = df["title"].align(items.set_index("item_id")["title"])
    assert np.all(ds_ts == orig_ts)

    assert ds.entities("item").attribute("title").is_scalar
    df = ds.entities("item").attribute("title").pandas()
    assert isinstance(df, pd.Series)
    ds_ts, orig_ts = df.align(items.set_index("item_id")["title"])
    assert np.all(ds_ts == orig_ts)


def test_item_scalar_series_arrays():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})

    dsb.add_entities("item", items["item_id"].values)
    dsb.add_scalar_attribute("item", "title", items["item_id"], items["title"])
    sa = dsb.schema.entities["item"].attributes["title"]
    assert sa.layout == AttrLayout.SCALAR

    ds = dsb.build()

    df = ds.entities("item").pandas().set_index("item_id")
    assert "title" in df.columns
    ds_ts, orig_ts = df["title"].align(items.set_index("item_id")["title"])
    assert np.all(ds_ts == orig_ts)

    assert ds.entities("item").attribute("title").is_scalar
    df = ds.entities("item").attribute("title").pandas()
    assert isinstance(df, pd.Series)
    ds_ts, orig_ts = df.align(items.set_index("item_id")["title"])
    assert np.all(ds_ts == orig_ts)


@mark.xfail(reason="attributes at insert time not yet implemented")
def test_item_insert_scalar_df():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})

    dsb.add_entities("item", items[["item_id", "title"]])
    sa = dsb.schema.entities["item"].attributes["title"]
    assert sa.layout == AttrLayout.SCALAR

    ds = dsb.build()

    df = ds.entities("item").pandas().set_index("item_id")
    assert "title" in df.columns
    ds_ts, orig_ts = df["title"].align(items.set_index("item_id")["title"])
    assert np.all(ds_ts == orig_ts)

    assert ds.entities("item").attribute("title").is_scalar
    df = ds.entities("item").attribute("title").pandas()
    assert isinstance(df, pd.Series)
    ds_ts, orig_ts = df.align(items.set_index("item_id")["title"])
    assert np.all(ds_ts == orig_ts)


@mark.xfail(reason="attributes at insert time not yet implemented")
def test_item_update_titles():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})

    dsb.add_entities("item", items)

    dsb.add_scalar_attribute(
        "item", "title", pd.Series({2: "Board Game Adventure", 110: "Briarheart"})
    )

    ds = dsb.build()
    df = ds.entities("item").attribute("title").pandas()
    assert df.loc[1, "title"] == "Toy Story (1995)"
    assert df.loc[2, "title"] == "Board Game Adventure"
    assert df.loc[110, "title"] == "Briarheart"


def test_item_list_series():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"}).set_index("item_id")

    genres = items["genres"].str.split("|")
    g_counts = genres.apply(len)

    dsb.add_entities("item", items.index.values)
    dsb.add_list_attribute("item", "genres", genres)
    va = dsb.schema.entities["item"].attributes["genres"]
    assert va.layout == AttrLayout.LIST

    ds = dsb.build()

    assert ds.entities("item").attribute("genres").is_list
    arr = ds.entities("item").attribute("genres").arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    assert isinstance(arr, pa.ListArray)
    assert len(arr) == len(genres)
    assert np.all(
        arr.value_lengths().fill_null(0).to_numpy()
        == g_counts.reindex(ds.entities("item").ids()).values
    )

    gs = ds.entities("item").attribute("genres").pandas()
    assert np.all(gs.index == ds.items.ids())
    gs, gs2 = gs.align(genres, join="outer")
    assert np.all(gs.notnull())
    assert np.all(gs2.notnull())
    gs = gs.apply(lambda gl: ",".join(sorted(gl)))
    gs2 = gs2.apply(lambda gl: ",".join(sorted(gl)))
    assert np.all(gs == gs2)


def test_item_list_random():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"}).set_index("item_id")

    genres = items["genres"].str.split("|")

    genres = genres.sample(n=200)

    dsb.add_entities("item", items.index.values)
    dsb.add_list_attribute("item", "genres", genres.index.values, genres.tolist())

    va = dsb.schema.entities["item"].attributes["genres"]
    assert va.layout == AttrLayout.LIST

    ds = dsb.build()

    assert ds.entities("item").attribute("genres").is_list

    gs = ds.entities("item").attribute("genres").pandas()

    gcomp, gscomp = genres.align(gs, "left")
    assert np.all(gcomp.apply(len) == gscomp.apply(len))


def test_item_list_df():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})
    items["genres"] = items["genres"].str.split("|")
    g_counts = items.set_index("item_id")["genres"].apply(len)

    dsb.add_entities("item", items["item_id"].values)
    dsb.add_list_attribute("item", "genres", items[["item_id", "genres"]])
    va = dsb.schema.entities["item"].attributes["genres"]
    assert va.layout == AttrLayout.LIST

    ds = dsb.build()
    assert ds.entities("item").attribute("genres").is_list
    arr = ds.entities("item").attribute("genres").arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    assert isinstance(arr, pa.ListArray)
    assert len(arr) == len(items)
    assert np.all(
        arr.value_lengths().fill_null(0).to_numpy()
        == g_counts.reindex(ds.entities("item").ids()).values
    )


@mark.xfail(reason="attributes at insert time not yet implemented")
def test_item_initial_list_df():
    dsb = DatasetBuilder()

    items = pd.read_csv(ml_test_dir / "movies.csv")
    items = items.rename(columns={"movieId": "item_id"})
    items["genres"] = items["genres"].str.split("|")

    dsb.add_entities("item", items)
    va = dsb.schema.entities["item"].attributes["genres"]
    assert va.layout == AttrLayout.LIST

    ds = dsb.build()

    assert ds.entities("item").attribute("genres").is_list

    arr = ds.entities("item").attribute("title").arrow()
    assert isinstance(arr, pa.StringArray)
    assert len(arr)

    arr = ds.entities("item").attribute("genres").arrow()
    assert isinstance(arr, pa.ListArray)
    assert len(arr) == len(items)


def test_item_vector(rng: np.random.Generator, ml_ratings: pd.DataFrame):
    dsb = DatasetBuilder()

    item_ids = ml_ratings["item_id"].unique()
    dsb.add_entities("item", item_ids)

    items = rng.choice(item_ids, 500, replace=False)
    vec = rng.standard_normal((500, 20))
    dsb.add_vector_attribute("item", "embedding", items, vec)
    va = dsb.schema.entities["item"].attributes["embedding"]
    assert va.layout == AttrLayout.VECTOR
    assert va.vector_size == 20

    tbl = dsb._tables["item"]
    assert tbl is not None
    col = tbl.column("embedding")
    assert len(col) == tbl.num_rows
    valid = col.is_valid().to_numpy(zero_copy_only=False)
    assert np.sum(valid) == 500
    col = col.combine_chunks()
    lens = col.value_lengths().fill_null(0).to_numpy()
    assert np.all(lens[valid] == 20)

    ds = dsb.build()
    assert ds.entities("item").attribute("embedding").is_vector
    assert ds.entities("item").attribute("embedding").names is None

    arr = ds.entities("item").attribute("embedding").arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    assert isinstance(arr, pa.FixedSizeListArray)
    assert np.sum(np.asarray(arr.is_valid())) == 500
    assert np.all(np.asarray(arr.value_lengths().drop_null()) == 20)

    arr = ds.entities("item").attribute("embedding").numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (len(item_ids), 20)
    assert np.sum(np.isfinite(arr)) == 500 * 20

    lim = ds.entities("item").attribute("embedding").drop_null()
    idx = pd.Index(lim.ids())
    v2 = lim.numpy()
    for i, id in enumerate(items):
        i2 = idx.get_loc(id)
        print(id, i, i2)
        assert np.all(vec[i, :] == v2[i2, :])

    arr = ds.entities("item").attribute("embedding").pandas(missing="omit")
    assert isinstance(arr, pd.DataFrame)
    assert np.all(arr.columns == np.arange(20))
    assert arr.shape == (500, 20)
    assert set(arr.index) == set(items)


def test_item_vector_names(rng: np.random.Generator, ml_ratings: pd.DataFrame):
    dsb = DatasetBuilder()

    item_ids = ml_ratings["item_id"].unique()
    dsb.add_entities("item", item_ids)

    items = rng.choice(item_ids, 500, replace=False)
    vec = rng.standard_normal((500, 5))
    dsb.add_vector_attribute("item", "embedding", items, vec, names=FRUITS)
    va = dsb.schema.entities["item"].attributes["embedding"]
    assert va.layout == AttrLayout.VECTOR

    ds = dsb.build()
    assert ds.entities("item").attribute("embedding").is_vector
    assert ds.entities("item").attribute("embedding").names == FRUITS

    arr = ds.entities("item").attribute("embedding").arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    assert isinstance(arr, pa.FixedSizeListArray)
    assert np.sum(np.asarray(arr.is_valid())) == 500
    assert np.all(np.asarray(arr.value_lengths().drop_null()) == 5)

    arr = ds.entities("item").attribute("embedding").numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (len(item_ids), 5)
    assert np.sum(np.isfinite(arr)) == 500 * 5

    arr = ds.entities("item").attribute("embedding").pandas(missing="omit")
    assert isinstance(arr, pd.DataFrame)
    assert np.all(arr.columns == FRUITS)
    assert arr.shape == (500, 5)
    assert set(arr.index) == set(items)


def test_item_vector_attr_subset(rng: np.random.Generator, ml_ratings: pd.DataFrame):
    dsb = DatasetBuilder()

    item_ids = ml_ratings["item_id"].unique()
    dsb.add_entities("item", item_ids)

    items = rng.choice(item_ids, 500, replace=False)
    vec = rng.standard_normal((500, 5))
    dsb.add_vector_attribute("item", "embedding", items, vec, names=FRUITS)
    va = dsb.schema.entities["item"].attributes["embedding"]
    assert va.layout == AttrLayout.VECTOR

    ds = dsb.build()

    query_items = set(rng.choice(items, 100, replace=False))
    query_items |= set(rng.choice(item_ids, 50, replace=False))
    query_items = np.array(list(query_items))
    q_known = np.isin(query_items, items)

    subset = ds.entities("item").select(ids=query_items)
    assert len(subset) == len(query_items)

    ss_attrs = subset.attribute("embedding")
    assert ss_attrs.is_vector
    assert ss_attrs.names == FRUITS

    assert np.all(ss_attrs.ids() == query_items)
    assert np.all(ss_attrs.numbers() == ds.items.numbers(query_items))

    arr = ss_attrs.arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    assert isinstance(arr, pa.FixedSizeListArray)
    assert len(arr) == len(query_items)
    assert np.all(arr.is_valid().to_numpy(zero_copy_only=False) == q_known)

    df = ss_attrs.pandas(missing="omit")
    assert isinstance(df, pd.DataFrame)
    assert np.all(df.columns == FRUITS)
    assert df.shape == (np.sum(q_known), 5)


def test_item_sparse_attribute(rng: np.random.Generator, ml_ratings: pd.DataFrame):
    dsb = DatasetBuilder()

    movies = pd.read_csv(ml_test_dir / "movies.csv")
    movies = movies.rename(columns={"movieId": "item_id"}).set_index("item_id")

    dsb.add_entities("item", movies.index)

    genre_lists = movies["genres"].str.split("|")
    genres = genre_lists.reset_index().explode("genres", ignore_index=True)
    gindex = pd.Index(np.unique(genres["genres"]))

    ids = movies.index.values.copy()
    rng.shuffle(ids)
    idx2 = pd.Index(ids)
    ig_rows = idx2.get_indexer_for(genres["item_id"])
    ig_cols = gindex.get_indexer_for(genres["genres"])
    ig_vals = np.ones(len(ig_rows), np.int32)

    arr = csr_array((ig_vals, (ig_rows, ig_cols)))
    dsb.add_vector_attribute("item", "genres", idx2, arr, names=gindex)

    ga = dsb.schema.entities["item"].attributes["genres"]
    assert ga.layout == AttrLayout.SPARSE

    ds = dsb.build()

    assert ds.entities("item").attribute("genres").is_sparse
    assert ds.entities("item").attribute("genres").names == gindex.values.tolist()

    mat = ds.entities("item").attribute("genres").scipy()
    assert isinstance(mat, csr_array)
    assert mat.nnz == arr.nnz

    m2 = ds.entities("item").select(ids=idx2.values).attribute("genres").scipy()
    assert isinstance(mat, csr_array)
    assert np.all(m2.indptr == arr.indptr)

    tensor = ds.entities("item").attribute("genres").torch()
    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse_csr
    assert len(tensor.values()) == arr.nnz

    arr = ds.entities("item").attribute("genres").arrow()
    assert pa.types.is_list(arr.type)

    for movie in rng.choice(movies.index.values, 50, replace=False):
        row = ds.items.number(movie)
        start = mat.indptr[row]
        end = mat.indptr[row + 1]
        cols = mat.indices[start:end]
        mgs = gindex[cols].tolist()
        assert mgs == genre_lists.loc[movie]
