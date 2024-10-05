# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle
from typing import Any

import numpy as np
import pandas as pd

from pytest import approx

import lenskit.util.test as lktu
from lenskit import util as lku
from lenskit.basic import BiasScorer
from lenskit.basic.composite import FallbackScorer
from lenskit.basic.history import KnownRatingScorer
from lenskit.data import Dataset
from lenskit.data.items import ItemList
from lenskit.data.types import EntityId
from lenskit.pipeline import Pipeline
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401


def test_fallback_fill_missing(ml_ds: Dataset):
    pipe = Pipeline()
    user = pipe.create_input("user", int)
    items = pipe.create_input("items")

    known = KnownRatingScorer()
    s1 = pipe.add_component("known", known, query=user, items=items)
    bias = BiasScorer()
    s2 = pipe.add_component("bias", bias, query=user, items=items)

    fallback = FallbackScorer()
    score = pipe.add_component("mix", fallback, scores=s1, backup=s2)

    pipe.train(ml_ds)

    # the first 2 of these are rated, the 3rd does not exist, and the other 2 are not rated
    items = [50, 17, 210, 1172, 2455]
    scored = pipe.run(score, user=2, items=ItemList(item_ids=items))

    assert len(scored) == len(items)
    assert np.all(scored.ids() == items)
    scores = scored.scores()
    assert scores is not None
    assert not np.any(np.isnan(scored.scores()))

    assert scores[:2] == approx(known(2, ItemList(item_ids=items[:2])).scores())
    assert scores[2:] == approx(bias(2, ItemList(item_ids=items[2:])).scores())
