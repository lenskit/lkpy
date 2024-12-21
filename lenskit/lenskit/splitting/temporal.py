import datetime as dt

import numpy as np

from lenskit.data import Dataset, from_interactions_df
from lenskit.data.collection import ItemListCollection

from .split import TTSplit


def split_global_time(data: Dataset, time: str | dt.datetime) -> TTSplit:
    if isinstance(time, str):
        time = dt.datetime.fromisoformat(time)

    matrix = data.interaction_log("pandas", fields="all", original_ids=True)
    if "timestamp" not in matrix:
        raise RuntimeError("temporal split requires timestamp")

    ts_col = matrix["timestamp"]
    ts_col = np.asarray(ts_col)

    if np.isdtype(ts_col.dtype, "numeric"):
        mask = ts_col >= time.timestamp()
    else:
        mask = ts_col >= time

    train = matrix[~mask]
    test = matrix[mask]

    train_ds = from_interactions_df(train)
    test_ilc = ItemListCollection.from_df(test, ["user_id"])
    return TTSplit(train_ds, test_ilc)
