import datetime as dt

import numpy as np

from lenskit.data import Dataset
from lenskit.splitting import split_global_time


def test_temporal_split(ml_ds: Dataset):
    point = dt.datetime.fromisoformat("2015-01-01")

    split = split_global_time(ml_ds, "2015-01-01")

    n_test = sum(len(il) for il in split.test.lists())
    assert n_test + split.train.interaction_count == ml_ds.interaction_count

    assert np.all(split.train.interaction_log("pandas")["timestamp"] < point.timestamp())
    for u, il in split.test:
        ts = il.field("timestamp")
        assert ts is not None
        assert np.all(ts >= point.timestamp())
