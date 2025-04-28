# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from lenskit.basic.candidates import (
    AllTrainingItemsCandidateSelector,
    UnratedTrainingItemsCandidateSelector,
)
from lenskit.data import Dataset, from_interactions_df
from lenskit.testing import ml_ds, ml_ratings  # noqa: F401


def test_all(ml_ds: Dataset):
    sel = AllTrainingItemsCandidateSelector()
    sel.train(ml_ds)

    cands = sel()
    assert len(cands) == ml_ds.item_count
    assert np.all(cands.ids() == ml_ds.items.ids())


def test_unrated_selector(ml_ds: Dataset):
    sel = UnratedTrainingItemsCandidateSelector()
    sel.train(ml_ds)

    row = ml_ds.user_row(100)
    assert row is not None
    cands = sel(query=row)

    assert len(cands) <= ml_ds.item_count
    assert len(cands) == len(set(ml_ds.items.ids()) - set(row.ids()))
