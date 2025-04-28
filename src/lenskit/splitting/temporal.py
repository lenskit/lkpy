# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import datetime as dt
from typing import Sequence, overload

from lenskit.data import Dataset, DatasetBuilder, ItemListCollection
from lenskit.logging import get_logger

from .split import TTSplit

_log = get_logger(__name__)


@overload
def split_global_time(
    data: Dataset,
    time: int | float | str | dt.datetime,
    end: int | float | str | dt.datetime | None = None,
    filter_test_users: bool = False,
) -> TTSplit: ...
@overload
def split_global_time(
    data: Dataset,
    time: Sequence[int | float | str | dt.datetime],
    end: int | float | str | dt.datetime | None = None,
    filter_test_users: bool = False,
) -> list[TTSplit]: ...
def split_global_time(
    data: Dataset,
    time: int | float | str | dt.datetime | Sequence[int | float | str | dt.datetime],
    end: int | float | str | dt.datetime | None = None,
    filter_test_users: bool = False,
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
        end:
            A final cutoff time for the testing data.
        filter_test_users:
            Limit test data to only have users who had item in the training data.

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
        tlog.debug("building training data set")
        train_ds = train_build.build()

        t2 = end
        if i + 1 < len(times):
            t2 = times[i + 1]

        if t2 is None:
            test = matrix[mask]
        else:
            tlog = tlog.bind(test_end=t2)
            tlog.debug("filtering test data for upper bound")
            test = matrix[mask & (ts_col < t2)]

        if filter_test_users:
            user_data = train_ds.user_stats()
            train_users = user_data.index[user_data["count"] > 0]
            test = test[test["user_id"].isin(train_users)]

        tlog.debug("building testing item lists")
        test_ilc = ItemListCollection.from_df(test, ["user_id"])
        tlog.debug("built split with %d train interactions", train_ds.interaction_count)
        results.append(TTSplit(train_ds, test_ilc))

    if rv == "sequence":
        return results
    else:
        assert len(results) == 1
        return results[0]


def split_temporal_fraction(
    data: Dataset, test_fraction: float, filter_test_users: bool = False
) -> TTSplit:
    """
    Do a global temporal split of a data set based on a test set size.

    Args:
        data:
            The dataset to split.
        test_fraction:
            The fraction of the interactions to put in the testing data.
        filter_test_users:
            Limit test data to only have users who had item in the training data.
    """
    df = data.interaction_table(format="pandas")
    point = df["timestamp"].quantile(1 - test_fraction)
    return split_global_time(data, point, filter_test_users=filter_test_users)


def _make_time(t: int | float | str | dt.datetime) -> dt.datetime:
    if isinstance(t, (int, float)):
        return dt.datetime.fromtimestamp(t)
    elif isinstance(t, str):
        return dt.datetime.fromisoformat(t)
    else:
        return t
