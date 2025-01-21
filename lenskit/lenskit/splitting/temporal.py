import datetime as dt
from typing import Sequence, overload

import numpy as np

from lenskit.data import Dataset, DatasetBuilder, ItemListCollection

from .split import TTSplit


@overload
def split_global_time(data: Dataset, time: str | dt.datetime) -> TTSplit: ...
@overload
def split_global_time(data: Dataset, time: Sequence[str | dt.datetime]) -> list[TTSplit]: ...
def split_global_time(
    data: Dataset, time: str | dt.datetime | Sequence[str | dt.datetime]
) -> TTSplit | list[TTSplit]:
    """
    Global temporal train-test split.  This splits a data set into train/test
    pairs using a single global timestamp.  When given multiple timestamps, it
    will return multiple splits, where split :math:`i` has training data from
    before :math:`t_i` and testing data on or after :math:`t_i` and before
    :math:`t_{i+1}` (the last split has no upper bound on the testing data).

    Stability:
        Caller

    Args:
        data:
            The dataset to split.
        time:
            Time or sequence of times at which to split.  Strings must be in ISO
            format.

    Returns:
        The data splits.
    """
    if isinstance(time, (str, int, float, dt.datetime)):
        times = [_make_time(time)]
        rv = "single"
    else:
        times = [_make_time(t) for t in time]
        rv = "sequence"

    iname = data.default_interaction_class()
    matrix = data.interactions().pandas(ids=True)
    if "timestamp" not in matrix:
        raise RuntimeError("temporal split requires timestamp")

    ts_col = matrix["timestamp"]
    ts_col = np.asarray(ts_col)

    if ts_col.dtype.kind in ("i", "u", "f"):
        times = [t.timestamp() for t in times]

    results = []
    for i, t in enumerate(times):
        mask = ts_col >= t
        train_build = DatasetBuilder(data)
        train_build.filter_interactions(iname, max_time=t)

        if i + 1 < len(times):
            test = matrix[mask & (ts_col < times[i + 1])]
        else:
            test = matrix[mask]

        train_ds = train_build.build()
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
