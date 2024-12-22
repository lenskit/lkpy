import datetime as dt
from typing import Sequence, overload

import numpy as np

from lenskit.data import Dataset, from_interactions_df
from lenskit.data.collection import ItemListCollection

from .split import TTSplit


@overload
def split_global_time(data: Dataset, time: str | dt.datetime) -> TTSplit: ...
@overload
def split_global_time(data: Dataset, time: Sequence[str | dt.datetime]) -> list[TTSplit]: ...
def split_global_time(
    data: Dataset, time: str | dt.datetime | Sequence[str | dt.datetime]
) -> TTSplit | list[TTSplit]:
    if isinstance(time, (str, int, float, dt.datetime)):
        times = [_make_time(time)]
        rv = "single"
    else:
        times = [_make_time(t) for t in time]
        rv = "sequence"

    matrix = data.interaction_log("pandas", fields="all", original_ids=True)
    if "timestamp" not in matrix:
        raise RuntimeError("temporal split requires timestamp")

    ts_col = matrix["timestamp"]
    ts_col = np.asarray(ts_col)

    if ts_col.dtype.kind in ("i", "u", "f"):
        times = [t.timestamp() for t in times]

    results = []
    for i, t in enumerate(times):
        mask = ts_col >= t

        train = matrix[~mask]
        if i + 1 < len(times):
            test = matrix[mask & (ts_col < times[i + 1])]
        else:
            test = matrix[mask]

        train_ds = from_interactions_df(train)
        test_ilc = ItemListCollection.from_df(test, ["user_id"])
        results.append(TTSplit(train_ds, test_ilc))

    if rv == "sequence":
        return results
    else:
        assert len(results) == 1
        return results[0]


def _make_time(t: int | float | str | dt.datetime) -> dt.datetime:
    if isinstance(t, (int, float)):
        return dt.datetime.fromtimestamp(t)
    elif isinstance(t, str):
        return dt.datetime.fromisoformat(t)
    else:
        return t
