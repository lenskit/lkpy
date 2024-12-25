# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd

from pytest import approx

from lenskit.basic import BiasScorer
from lenskit.basic.composite import FallbackScorer
from lenskit.basic.history import KnownRatingScorer
from lenskit.data import Dataset
from lenskit.data.items import ItemList
from lenskit.data.types import ID
from lenskit.operations import predict, score
from lenskit.pipeline import Pipeline
from lenskit.pipeline.common import RecPipelineBuilder

_log = logging.getLogger(__name__)


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


def test_fallback_double_bias(rng: np.random.Generator, ml_ds: Dataset):
    builder = RecPipelineBuilder()
    builder.scorer(BiasScorer(damping=50))
    builder.predicts_ratings(fallback=BiasScorer(damping=0))
    pipe = builder.build("double-bias")

    _log.info("pipeline configuration: %s", pipe.get_config().model_dump_json(indent=2))

    pipe.train(ml_ds)

    for user in rng.choice(ml_ds.users.ids(), 100):
        items = rng.choice(ml_ds.items.ids(), 500)
        scores = score(pipe, user, items)
        scores = scores.scores()
        assert scores is not None
        assert not np.any(np.isnan(scores))

        preds = predict(pipe, user, items)

        preds = preds.scores()
        assert preds is not None
        assert not np.any(np.isnan(preds))

        assert scores == approx(preds)
