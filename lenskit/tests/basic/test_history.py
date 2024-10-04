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

import lenskit.util.test as lktu
from lenskit.basic import PopScorer
from lenskit.basic.history import UserTrainingHistoryLookup
from lenskit.basic.topn import TopNRanker
from lenskit.data.items import ItemList
from lenskit.data.query import RecQuery
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401

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


def test_lookup_items_only(ml_ds):
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
