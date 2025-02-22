# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: basic
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa

from pytest import approx, mark, raises, warns

from lenskit.data import Dataset, DatasetBuilder
from lenskit.data.schema import AllowableTroolean
from lenskit.diagnostics import DataError, DataWarning


@dataclass
class QueryDT:
    s: str
    dt: datetime
    ts: float
    thresh: str | datetime | float
    compare: datetime | float

    @classmethod
    def create(
        cls,
        spec: str,
        t_fmt: Literal["datetime", "timestamp", "string"],
        d_fmt: Literal["int", "timestamp"],
    ) -> QueryDT:
        dt = datetime.fromisoformat(spec)
        ts = dt.timestamp()
        match t_fmt:
            case "datetime":
                thresh = dt
            case "timestamp":
                thresh = ts
            case "string":
                thresh = spec

        match d_fmt:
            case "int":
                comp = ts
            case "timestamp":
                comp = dt

        return cls(s=spec, dt=dt, ts=ts, thresh=thresh, compare=comp)


@mark.parametrize(
    ["ts_fmt", "q_fmt"], product(["int", "timestamp"], ["datetime", "timestamp", "string"])
)
def test_filter_ratings_min_time(
    ts_fmt: Literal["int", "timestamp"],
    q_fmt: Literal["datetime", "timestamp", "string"],
    ml_ratings: pd.DataFrame,
):
    dsb = DatasetBuilder()
    if ts_fmt == "int":
        ml_ratings = ml_ratings.assign(
            timestamp=(ml_ratings["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        )
    dsb.add_interactions(
        "rating", ml_ratings, entities=["user", "item"], missing="insert", default=True
    )
    q = QueryDT.create("2001-01-01", q_fmt, ts_fmt)
    dsb.filter_interactions("rating", min_time=q.thresh)
    ds = dsb.build()
    assert ds.interactions().pandas()["timestamp"].min() >= q.compare
    assert ds.interactions().pandas()["timestamp"].max() == ml_ratings["timestamp"].max()


@mark.parametrize(
    ["ts_fmt", "q_fmt"], product(["int", "timestamp"], ["datetime", "timestamp", "string"])
)
def test_filter_ratings_max_time(
    ts_fmt: Literal["int", "timestamp"],
    q_fmt: Literal["datetime", "timestamp", "string"],
    ml_ratings: pd.DataFrame,
):
    dsb = DatasetBuilder()
    if ts_fmt == "int":
        ml_ratings = ml_ratings.assign(
            timestamp=(ml_ratings["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        )
    dsb.add_interactions(
        "rating", ml_ratings, entities=["user", "item"], missing="insert", default=True
    )
    q = QueryDT.create("2001-01-01", q_fmt, ts_fmt)
    dsb.filter_interactions("rating", max_time=q.thresh)
    ds = dsb.build()
    assert ds.interactions().pandas()["timestamp"].min() == ml_ratings["timestamp"].min()
    assert ds.interactions().pandas()["timestamp"].max() < q.compare


@mark.parametrize(
    ["ts_fmt", "q_fmt"], product(["int", "timestamp"], ["datetime", "timestamp", "string"])
)
def test_filter_ratings_min_max_time(
    ts_fmt: Literal["int", "timestamp"],
    q_fmt: Literal["datetime", "timestamp", "string"],
    ml_ratings: pd.DataFrame,
):
    dsb = DatasetBuilder()
    if ts_fmt == "int":
        ml_ratings = ml_ratings.assign(
            timestamp=(ml_ratings["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        )
    dsb.add_interactions(
        "rating", ml_ratings, entities=["user", "item"], missing="insert", default=True
    )
    q1 = QueryDT.create("2001-01-01", q_fmt, ts_fmt)
    q2 = QueryDT.create("2004-01-01", q_fmt, ts_fmt)
    dsb.filter_interactions("rating", min_time=q1.thresh, max_time=q2.thresh)
    ds = dsb.build()
    assert ds.interactions().pandas()["timestamp"].min() >= q1.compare
    assert ds.interactions().pandas()["timestamp"].max() < q2.compare


def test_filter_ratings_nums_df(rng, ml_ds: Dataset):
    dsb = DatasetBuilder(ml_ds)

    rates = ml_ds.interactions().pandas(attributes=[])
    samp = rates.sample(1000, replace=False, random_state=rng, ignore_index=True)

    dsb.filter_interactions("rating", remove=samp)

    ds = dsb.build()
    assert ds.interaction_count == ml_ds.interaction_count - 1000

    orig_istats = ml_ds.item_stats()
    new_istats = ds.item_stats()
    si_counts = samp["item_num"].value_counts()
    si_counts = pd.Series(si_counts.values, index=ml_ds.items.ids(si_counts.index.values))
    si_counts = si_counts.reindex(new_istats.index, fill_value=0)

    assert np.all(new_istats["count"] == orig_istats["count"] - si_counts)


def test_filter_ratings_nums_tbl(rng, ml_ds: Dataset):
    dsb = DatasetBuilder(ml_ds)

    rates = ml_ds.interactions().pandas(attributes=[])
    samp_df = rates.sample(1000, replace=False, random_state=rng, ignore_index=True)
    samp = pa.Table.from_pandas(samp_df, preserve_index=False)

    dsb.filter_interactions("rating", remove=samp)

    ds = dsb.build()
    assert ds.interaction_count == ml_ds.interaction_count - 1000

    orig_istats = ml_ds.item_stats()
    new_istats = ds.item_stats()
    si_counts = samp_df["item_num"].value_counts()
    si_counts = pd.Series(si_counts.values, index=ml_ds.items.ids(si_counts.index.values))
    si_counts = si_counts.reindex(new_istats.index, fill_value=0)

    assert np.all(new_istats["count"] == orig_istats["count"] - si_counts)


def test_filter_ratings_ids_df(rng, ml_ds: Dataset):
    dsb = DatasetBuilder(ml_ds)

    rates = ml_ds.interactions().pandas(attributes=[], ids=True)
    samp = rates.sample(1000, replace=False, random_state=rng, ignore_index=True)

    dsb.filter_interactions("rating", remove=samp)

    ds = dsb.build()
    assert ds.interaction_count == ml_ds.interaction_count - 1000

    orig_istats = ml_ds.item_stats()
    new_istats = ds.item_stats()
    si_counts = samp["item_id"].value_counts()
    si_counts = si_counts.reindex(new_istats.index, fill_value=0)

    assert np.all(new_istats["count"] == orig_istats["count"] - si_counts)


def test_filter_ratings_ids_tbl(rng, ml_ds: Dataset):
    dsb = DatasetBuilder(ml_ds)

    rates = ml_ds.interactions().pandas(attributes=[], ids=True)
    samp_df = rates.sample(1000, replace=False, random_state=rng, ignore_index=True)
    samp = pa.Table.from_pandas(samp_df, preserve_index=False)

    dsb.filter_interactions("rating", remove=samp)

    ds = dsb.build()
    assert ds.interaction_count == ml_ds.interaction_count - 1000

    orig_istats = ml_ds.item_stats()
    new_istats = ds.item_stats()
    si_counts = samp_df["item_id"].value_counts()
    si_counts = si_counts.reindex(new_istats.index, fill_value=0)

    assert np.all(new_istats["count"] == orig_istats["count"] - si_counts)
