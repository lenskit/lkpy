# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Whole-pipeline integration tests to check for composite bugs.
"""

import numpy as np

from lenskit import batch
from lenskit.basic import (
    RandomSelector,
    UnratedTrainingItemsCandidateSelector,
    UserTrainingHistoryLookup,
)
from lenskit.data import Dataset, QueryInput
from lenskit.pipeline import PipelineBuilder


def test_training_items_removed(rng: np.random.Generator, ml_ds: Dataset):
    pb = PipelineBuilder()
    query = pb.create_input("query", QueryInput)
    query = pb.add_component("lookup", UserTrainingHistoryLookup, query=query)
    candidates = pb.add_component("candidate-selector", UnratedTrainingItemsCandidateSelector)
    pb.add_component("recommender", RandomSelector, {"n": 100}, items=candidates)

    pipe = pb.build()
    pipe.train(ml_ds)

    users = rng.choice(ml_ds.users.ids(), 200)
    recs = batch.recommend(pipe, users)

    for key, items in recs.items():
        uid = key.user_id
        assert uid in users
        train_items = ml_ds.user_row(uid)
        assert train_items is not None

        assert len(items) == 100
        assert not np.any(np.isin(items.ids(), train_items.ids()))
        assert not np.any(np.isin(items.numbers(), train_items.numbers()))
