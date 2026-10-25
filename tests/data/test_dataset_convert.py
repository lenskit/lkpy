# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pyarrow as pa

from lenskit.data import from_interactions_df


def test_from_interactions_df_forwards_timestamp_unit():
    data = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "time": [1_700_000_000_123, 1_700_000_000_456],
        }
    )

    ds = from_interactions_df(data, timestamp_col="time", timestamp_unit="ms")

    log = ds.interaction_table(format="arrow")
    assert log.field("timestamp").type == pa.timestamp("ms")


def test_item_subset(rng: np.random.Generator, ml_ratings: pd.DataFrame):
    items = rng.choice(ml_ratings["item_id"].unique(), 500, replace=False)
    ds = from_interactions_df(ml_ratings, items=items)

    assert ds.interaction_count < len(ml_ratings)
    assert ds.item_count == len(items)
    assert np.all(ds.items.ids() == np.sort(items))

    log = ds.interaction_table(format="pandas", original_ids=True)
    assert np.all(log["item_id"].isin(items))


def test_user_subset(rng: np.random.Generator, ml_ratings: pd.DataFrame):
    users = rng.choice(ml_ratings["user_id"].unique(), 500, replace=False)
    ds = from_interactions_df(ml_ratings, users=users)

    assert ds.interaction_count < len(ml_ratings)
    assert ds.user_count == len(users)
    assert np.all(ds.users.ids() == np.sort(users))

    log = ds.interaction_table(format="pandas", original_ids=True)
    assert np.all(log["user_id"].isin(users))
