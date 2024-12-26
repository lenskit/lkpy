import numpy as np
import pandas as pd

from lenskit.data.convert import from_interactions_df


def test_item_subset(rng: np.random.Generator, ml_ratings: pd.DataFrame):
    items = rng.choice(ml_ratings["item_id"].unique(), 500, replace=False)
    ds = from_interactions_df(ml_ratings, items=items)

    assert ds.interaction_count < len(ml_ratings)
    assert ds.item_count == len(items)
    assert np.all(ds.items.ids() == np.sort(items))

    log = ds.interaction_log("pandas", original_ids=True)
    assert np.all(log["item_id"].isin(items))


def test_user_subset(rng: np.random.Generator, ml_ratings: pd.DataFrame):
    users = rng.choice(ml_ratings["user_id"].unique(), 500, replace=False)
    ds = from_interactions_df(ml_ratings, users=users)

    assert ds.interaction_count < len(ml_ratings)
    assert ds.user_count == len(users)
    assert np.all(ds.users.ids() == np.sort(users))

    log = ds.interaction_log("pandas", original_ids=True)
    assert np.all(log["user_id"].isin(users))
