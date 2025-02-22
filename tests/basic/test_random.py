# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle
from itertools import groupby

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import HealthCheck, assume, given, settings

from lenskit.basic.random import RandomSelector
from lenskit.data.items import ItemList
from lenskit.testing import scored_lists

_log = logging.getLogger(__name__)


@given(scored_lists())
def test_unlimited_selection(items: ItemList):
    rsel = RandomSelector()
    ranked = rsel(items=items)

    ids = items.ids()

    assert len(ranked) == len(ids)
    assert set(ranked.ids()) == set(ids)


@given(st.integers(min_value=1, max_value=100), scored_lists())
def test_configured_truncation(n, items: ItemList):
    rsel = RandomSelector(n=n)
    ranked = rsel(items=items)

    assert len(ranked) == min(n, len(items))
    # all items are unique
    assert len(set(ranked.ids())) == len(ranked)


@given(st.integers(min_value=1, max_value=100), scored_lists())
def test_runtime_truncation(n, items: ItemList):
    rsel = RandomSelector()
    ranked = rsel(items=items, n=n)

    assert len(ranked) == min(n, len(items))
    # all items are unique
    assert len(set(ranked.ids())) == len(ranked)


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(min_value=5, max_value=100),
    st.lists(st.emails(), min_size=5, unique=True),
    scored_lists(n=(5, 500)),
)
def test_random_seeding(n: int, users: list[str], items: ItemList):
    assume(n < len(items))
    rsel = RandomSelector(rng=(42, "user"))

    distinct = set()

    for user in users:
        ranked = rsel(items=items, n=n, query=user)

        assert len(ranked) == n
        # all items are unique
        assert len(set(ranked.ids())) == len(ranked)

        # ranking again gets the same list
        r2 = rsel(items=items, n=n, query=user)
        assert np.all(r2.ids() == ranked.ids())

        distinct |= set(ranked.ids())

    # we should have more than n distinct values (randomness works)
    assert len(distinct) > n
