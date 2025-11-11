# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from lenskit.basic.candidates import (
    AllTrainingItemsCandidateSelector,
    TrainingItemsCandidateSelector,
    UnratedTrainingItemsCandidateSelector,
)
from lenskit.data import Dataset, ItemList, RecQuery
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


def test_unrated_selector_many(rng: np.random.Generator, ml_ds: Dataset):
    sel = UnratedTrainingItemsCandidateSelector()
    sel.train(ml_ds)

    for u in rng.choice(ml_ds.users.ids(), 200):
        row = ml_ds.user_row(100)
        assert row is not None
        cands = sel(query=row)

        assert len(cands) <= ml_ds.item_count
        assert len(cands) == len(set(ml_ds.items.ids()) - set(row.ids()))
        assert not np.any(np.isin(cands.ids(), row.ids()))
        assert not np.any(np.isin(cands.numbers(), row.numbers()))


def test_training_selector_history(ml_ds: Dataset):
    sel = TrainingItemsCandidateSelector()
    sel.train(ml_ds)

    row = ml_ds.user_row(100)
    assert row is not None
    cands = sel(query=row)

    assert len(cands) <= ml_ds.item_count
    assert len(cands) == len(set(ml_ds.items.ids()) - set(row.ids()))


def test_training_selector_all(ml_ds: Dataset):
    sel = TrainingItemsCandidateSelector(exclude=None)
    sel.train(ml_ds)

    row = ml_ds.user_row(100)
    assert row is not None
    cands = sel(query=row)

    assert len(cands) == ml_ds.item_count


def test_training_selector_session_empty(ml_ds: Dataset):
    sel = TrainingItemsCandidateSelector(exclude="session")
    sel.train(ml_ds)

    row = ml_ds.user_row(100)
    assert row is not None
    cands = sel(query=row)

    assert len(cands) == ml_ds.item_count


def test_training_selector_session(ml_ds: Dataset):
    sel = TrainingItemsCandidateSelector(exclude="session")
    sel.train(ml_ds)

    row = ml_ds.user_row(100)
    session = ItemList(item_nums=np.arange(0, 10, 2), vocabulary=ml_ds.items)
    assert row is not None
    cands = sel(query=RecQuery(100, history_items=row, session_items=session))

    assert len(cands) <= ml_ds.item_count
    assert len(cands) == len(set(ml_ds.items.ids()) - set(session.ids()))


def test_training_selector_multi(ml_ds: Dataset):
    sel = TrainingItemsCandidateSelector(exclude=["session", "history"])
    sel.train(ml_ds)

    row = ml_ds.user_row(100)
    session = ItemList(item_nums=np.arange(0, 10, 2), vocabulary=ml_ds.items)
    assert row is not None
    cands = sel(query=RecQuery(100, history_items=row, session_items=session))

    assert len(cands) <= ml_ds.item_count
    assert len(cands) == len(set(ml_ds.items.ids()) - set(session.ids()) - set(row.ids()))
