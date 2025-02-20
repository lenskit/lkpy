# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given, settings

from lenskit.basic import PopScorer
from lenskit.basic.history import KnownRatingScorer, UserTrainingHistoryLookup
from lenskit.basic.topn import TopNRanker
from lenskit.data import Dataset, ItemList, RecQuery

_log = logging.getLogger(__name__)


def test_lookup_user_id(ml_ds):
    lookup = UserTrainingHistoryLookup()
    lookup.train(ml_ds)

    user = ml_ds.users.id(100)
    ds_row = ml_ds.user_row(user)
    query = lookup(user)

    assert isinstance(query, RecQuery)
    assert query.user_id == user
    assert query.user_items is not None
    assert len(query.user_items) == len(ds_row)
    assert np.all(query.user_items.ids() == ds_row.ids())


def test_lookup_no_override(ml_ds):
    "ensure history does not override when items already present"
    lookup = UserTrainingHistoryLookup()
    lookup.train(ml_ds)

    user = ml_ds.users.id(100)
    ds_row = ml_ds.user_row(user)
    query = RecQuery(user, ds_row[:-2])
    query = lookup(query)

    assert isinstance(query, RecQuery)
    assert query.user_id == user
    assert query.user_items is not None
    assert len(query.user_items) == len(ds_row) - 2
    assert np.all(query.user_items.ids() == ds_row[:-2].ids())


def test_lookup_items_only(ml_ds: Dataset):
    "ensure history accepts item list"
    lookup = UserTrainingHistoryLookup()
    lookup.train(ml_ds)

    user = ml_ds.users.id(100)
    ds_row = ml_ds.user_row(user)
    query = lookup(ds_row[:-5])

    assert isinstance(query, RecQuery)
    assert query.user_id is None
    assert query.user_items is not None
    assert len(query.user_items) == len(ds_row) - 5
    assert np.all(query.user_items.ids() == ds_row[:-5].ids())


def test_lookup_pickle(ml_ds: Dataset):
    "ensure we can correctly pickle a history component"
    lookup = UserTrainingHistoryLookup()
    lookup.train(ml_ds)

    blob = pickle.dumps(lookup)
    l2 = pickle.loads(blob)
    assert isinstance(l2, UserTrainingHistoryLookup)

    assert l2.interactions.count() == lookup.interactions.count()

    ds_row = ml_ds.user_row(user_id=100)
    l_row = lookup(100)
    l2_row = l2(100)

    assert l_row.user_id == 100
    assert np.all(l_row.user_items.ids() == ds_row.ids())
    assert l2_row.user_id == 100
    assert np.all(l2_row.user_items.ids() == ds_row.ids())


def test_known_rating_defaults(ml_ds: Dataset):
    algo = KnownRatingScorer()
    algo.train(ml_ds)

    # the first 2 of these are rated, the 3rd does not exist, and the other 2 are not rated
    items = [50, 17, 210, 1172, 2455]
    scored = algo(2, ItemList(item_ids=items))

    assert len(scored) == len(items)
    assert np.all(scored.ids() == items)

    scores = scored.scores()
    assert scores is not None
    assert np.all(np.isnan(scores[-3:]))
    assert not np.any(np.isnan(scores[:-3]))

    row = ml_ds.user_row(2)
    assert row is not None
    udf = row.field("rating", "pandas", index="ids")
    assert udf is not None
    assert np.all(scores[~np.isnan(scores)] == udf.loc[[50, 17]].values)


def test_known_indicator(ml_ds: Dataset):
    algo = KnownRatingScorer(score="indicator")
    algo.train(ml_ds)

    # the first 2 of these are rated, the 3rd does not exist, and the other 2 are not rated
    items = [50, 17, 210, 1172, 2455]
    scored = algo(2, ItemList(item_ids=items))

    assert len(scored) == len(items)
    assert np.all(scored.ids() == items)

    scores = scored.scores()
    assert scores is not None
    assert np.all(scores == [1, 1, 0, 0, 0])


def test_known_query_ratings_none(ml_ds: Dataset):
    algo = KnownRatingScorer(score="rating", source="query")
    algo.train(ml_ds)

    # the first 2 of these are rated, the 3rd does not exist, and the other 2 are not rated
    items = [50, 17, 210, 1172, 2455]
    scored = algo(2, ItemList(item_ids=items))

    assert len(scored) == len(items)
    assert np.all(scored.ids() == items)

    # but no scores should be computed!
    scores = scored.scores()
    assert scores is not None
    assert np.all(np.isnan(scores))


def test_known_query_ratings(ml_ds: Dataset):
    algo = KnownRatingScorer(score="rating", source="query")
    algo.train(ml_ds)

    # the first 2 of these are rated, the 3rd does not exist, and the other 2 are not rated
    items = [50, 17, 210, 1172, 2455]
    scored = algo(ml_ds.user_row(2), ItemList(item_ids=items))

    assert len(scored) == len(items)
    assert np.all(scored.ids() == items)

    scores = scored.scores()
    assert scores is not None
    assert np.all(np.isnan(scores[-3:]))
    assert not np.any(np.isnan(scores[:-3]))

    row = ml_ds.user_row(2)
    assert row is not None
    udf = row.field("rating", "pandas", index="ids")
    assert udf is not None
    assert np.all(scores[~np.isnan(scores)] == udf.loc[[50, 17]].values)


def test_known_query_indicator(ml_ds: Dataset):
    algo = KnownRatingScorer(score="indicator", source="query")
    algo.train(ml_ds)

    # the first 2 of these are rated, the 3rd does not exist, and the other 2 are not rated
    items = [50, 17, 210, 1172, 2455]
    scored = algo(ml_ds.user_row(2), ItemList(item_ids=items))

    assert len(scored) == len(items)
    assert np.all(scored.ids() == items)

    scores = scored.scores()
    assert scores is not None
    assert np.all(scores == [1, 1, 0, 0, 0])


def test_known_query_default_indicator(ml_ds: Dataset):
    algo = KnownRatingScorer(source="query")
    algo.train(ml_ds)

    # the first 2 of these are rated, the 3rd does not exist, and the other 2 are not rated
    items = [50, 17, 210, 1172, 2455]
    scored = algo(ItemList(ml_ds.user_row(2), rating=False), ItemList(item_ids=items))

    assert len(scored) == len(items)
    assert np.all(scored.ids() == items)

    scores = scored.scores()
    assert scores is not None
    assert np.all(scores[:2] == [1, 1])
    assert np.all(np.isnan(scores[2:]))
