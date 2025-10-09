# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle

import pandas as pd

from lenskit.basic import PopScorer
from lenskit.data.items import ItemList
from lenskit.testing import BasicComponentTests, ScorerTests

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)


class TestPopScore(BasicComponentTests, ScorerTests):
    component = PopScorer


def test_popscore_quantile(rng, ml_ds):
    pop = PopScorer()
    pop.train(ml_ds)

    assert pop.item_scores_.max() == 1.0

    counts = ml_ds.item_stats()["count"]
    counts = counts.sort_values()

    winner = counts.index[-1]
    assert pop.item_scores_[ml_ds.items.number(winner)] == 1.0


def test_popscore_rank(rng, ml_ds):
    pop = PopScorer(score="rank")
    pop.train(ml_ds)

    counts = ml_ds.item_stats()["count"]
    counts = counts.sort_values()

    assert pop.item_scores_.max() == len(counts)

    winner = counts.index[-1]
    assert pop.item_scores_[ml_ds.items.number(winner)] == len(counts)


def test_popscore_counts(rng, ml_ds):
    pop = PopScorer(score="count")
    pop.train(ml_ds)

    counts = ml_ds.item_stats()["count"]

    scores, counts = pd.Series(pop.item_scores_, index=pop.items_.ids()).align(counts)
    assert all(scores == counts)

    items = rng.choice(counts.index, 100)
    scores = pop(ItemList(item_ids=items))
    assert all(scores.scores() == counts.loc[items])


def test_popscore_save_load(ml_ds):
    original = PopScorer()
    original.train(ml_ds)

    mod = pickle.dumps(original)
    pop = pickle.loads(mod)

    assert all(pop.item_scores_ == original.item_scores_)
