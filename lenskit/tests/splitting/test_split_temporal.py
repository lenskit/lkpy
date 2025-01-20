import datetime as dt

import numpy as np

from lenskit.data import Dataset
from lenskit.splitting import split_global_time


def test_temporal_split(ml_ds: Dataset):
    point = dt.datetime.fromisoformat("2015-01-01")

    split = split_global_time(ml_ds, "2015-01-01")

    n_test = sum(len(il) for il in split.test.lists())
    assert n_test + split.train.interaction_count == ml_ds.interaction_count

    assert np.all(split.train.interaction_log(format="pandas")["timestamp"] < point.timestamp())
    for u, il in split.test:
        ts = il.field("timestamp")
        assert ts is not None
        assert np.all(ts >= point.timestamp())


def test_multi_split(ml_ds: Dataset):
    p1 = dt.datetime.fromisoformat("2015-01-01")
    p2 = dt.datetime.fromisoformat("2016-01-01")

    valid, test = split_global_time(ml_ds, ["2015-01-01", "2016-01-01"])

    n_test = sum(len(il) for il in test.test.lists())
    n_valid = sum(len(il) for il in valid.test.lists())
    assert n_test + test.train.interaction_count == ml_ds.interaction_count
    assert n_test + n_valid + valid.train.interaction_count == ml_ds.interaction_count

    assert np.all(valid.train.interaction_log(format="pandas")["timestamp"] < p1.timestamp())
    assert np.all(test.train.interaction_log(format="pandas")["timestamp"] < p2.timestamp())
    for u, il in test.test:
        ts = il.field("timestamp")
        assert ts is not None
        assert np.all(ts >= p2.timestamp())

    for u, il in valid.test:
        ts = il.field("timestamp")
        assert ts is not None
        assert np.all(ts >= p1.timestamp())
        assert np.all(ts < p2.timestamp())
