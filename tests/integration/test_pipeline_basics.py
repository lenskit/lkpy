# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Whole-pipeline integration tests to check for composite bugs.
"""

import numpy as np

from pytest import fixture

from lenskit import batch, recommend
from lenskit.als import ImplicitMFScorer
from lenskit.basic import (
    RandomSelector,
    UnratedTrainingItemsCandidateSelector,
    UserTrainingHistoryLookup,
)
from lenskit.data import Dataset, ItemList, QueryInput, RecQuery
from lenskit.pipeline import Pipeline, PipelineBuilder, topn_pipeline


@fixture(scope="module")
def random_pipe(ml_ds: Dataset):
    pb = PipelineBuilder()
    query = pb.create_input("query", QueryInput)
    history = pb.add_component("history-lookup", UserTrainingHistoryLookup, query=query)
    candidates = pb.add_component(
        "candidate-selector", UnratedTrainingItemsCandidateSelector, query=history
    )
    pb.add_component("recommender", RandomSelector, {"n": 100}, items=candidates)

    pipe = pb.build()
    pipe.train(ml_ds)
    yield pipe


def test_history_correct(rng: np.random.Generator, ml_ds: Dataset, random_pipe: Pipeline):
    users = rng.choice(ml_ds.users.ids(), 200, replace=False)

    for uid in users:
        train_items = ml_ds.user_row(uid)
        assert train_items is not None

        query = random_pipe.run("history-lookup", query=uid)
        assert isinstance(query, RecQuery)
        assert query.user_id == uid
        user_items = query.user_items
        assert isinstance(user_items, ItemList)
        assert len(user_items) == len(train_items)
        assert set(user_items.ids()) == set(train_items.ids())
        assert len(user_items) == len(set(user_items.ids()))


def test_candidates_correct(rng: np.random.Generator, ml_ds: Dataset, random_pipe: Pipeline):
    users = rng.choice(ml_ds.users.ids(), 200, replace=False)

    for uid in users:
        train_items = ml_ds.user_row(uid)
        assert train_items is not None

        candidates = random_pipe.run("candidate-selector", query=uid)
        assert isinstance(candidates, ItemList)
        # make sure the candidates don't have any training items
        assert len(candidates) == ml_ds.item_count - len(train_items)
        assert len(candidates) == len(set(candidates.ids()))


def test_training_items_removed_solo(
    rng: np.random.Generator, ml_ds: Dataset, random_pipe: Pipeline
):
    users = rng.choice(ml_ds.users.ids(), 200, replace=False)

    for uid in users:
        items = recommend(random_pipe, uid)

        train_items = ml_ds.user_row(uid)
        assert train_items is not None

        assert len(items) == 100
        assert not np.any(np.isin(items.ids(), train_items.ids()))
        assert not np.any(np.isin(items.numbers(), train_items.numbers()))
        assert len(items) == len(set(items.ids()))


def test_training_items_removed_batch(
    rng: np.random.Generator, ml_ds: Dataset, random_pipe: Pipeline
):
    users = rng.choice(ml_ds.users.ids(), 200, replace=False)
    recs = batch.recommend(random_pipe, users, n_jobs=1)

    for key, items in recs.items():
        uid = key.user_id
        assert uid in users
        train_items = ml_ds.user_row(uid)
        assert train_items is not None

        assert len(items) == 100
        # make sure we didn't recommend any training items
        assert not np.any(np.isin(items.ids(), train_items.ids()))
        assert not np.any(np.isin(items.numbers(), train_items.numbers()))
        assert len(items) == len(set(items.ids()))


def test_training_items_removed_scorer(rng: np.random.Generator, ml_ds: Dataset):
    pipe = topn_pipeline(scorer=ImplicitMFScorer, n=100)
    pipe.train(ml_ds)

    users = rng.choice(ml_ds.users.ids(), 200, replace=False)
    recs = batch.recommend(pipe, users, n_jobs=1)

    for key, items in recs.items():
        uid = key.user_id
        assert uid in users
        train_items = ml_ds.user_row(uid)
        assert train_items is not None

        assert len(items) == 100
        # make sure we didn't recommend any training items
        assert not np.any(np.isin(items.ids(), train_items.ids()))
        assert not np.any(np.isin(items.numbers(), train_items.numbers()))
        assert len(items) == len(set(items.ids()))
