import datetime as dt
from typing import Sequence, overload

from lenskit.data import Dataset, DatasetBuilder, ItemListCollection
from lenskit.logging import get_logger

from .split import TTSplit

_log = get_logger(__name__)


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
    log = _log.bind(n_records=data.interaction_count)
    if isinstance(time, (str, int, float, dt.datetime)):
        times = [_make_time(time)]
        rv = "single"
    else:
        times = [_make_time(t) for t in time]
        rv = "sequence"
        log = log.bind(n_splits=len(times))

    iname = data.default_interaction_class()
    matrix = data.interactions().pandas(ids=True)
    if "timestamp" not in matrix:
        raise RuntimeError("temporal split requires timestamp")

    ts_col = matrix["timestamp"]
    # ts_col = np.asarray(ts_col)

    if ts_col.dtype.kind in ("i", "u", "f"):
        log.debug("converting query timestamps")
        times = [t.timestamp() for t in times]

    results = []
    for i, t in enumerate(times):
        tlog = log.bind(number=i, test_start=t)
        tlog.debug("creating initial split")
        mask = ts_col >= t
        train_build = DatasetBuilder(data)
        train_build.filter_interactions(iname, max_time=t)

        if i + 1 < len(times):
            t2 = times[i + 1]
            tlog = tlog.bind(test_end=t2)
            tlog.debug("filtering test data for upper bound")
            test = matrix[mask & (ts_col < t2)]
        else:
            test = matrix[mask]

        tlog.debug("building training data set")
        train_ds = train_build.build()
        tlog.debug("building testing item lists")
        test_ilc = ItemListCollection.from_df(test, ["user_id"])
        tlog.debug("built split with %d train interactions", train_ds.interaction_count)
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
