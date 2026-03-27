# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle

from lenskit.data import Dataset


def test_dataset_pickle(ml_ds: Dataset):
    data = pickle.dumps(ml_ds)

    ds = pickle.loads(data)
    assert isinstance(ds, Dataset)

    assert ds.item_count == ml_ds.item_count
    assert ds.user_count == ml_ds.user_count
    assert ds.interaction_count == ml_ds.interaction_count
