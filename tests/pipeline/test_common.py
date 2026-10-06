# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from pytest import fixture, mark

from lenskit.basic.history import UserTrainingHistoryLookup
from lenskit.data import ItemList, QueryInput, RecQuery, from_interactions_df
from lenskit.pipeline import Pipeline, RecPipelineBuilder, predict_pipeline, topn_pipeline


def history_scorer(query: QueryInput, items: ItemList) -> ItemList:
    query = RecQuery.create(query)
    count = len(query.history_items) if query.history_items is not None else 0
    return ItemList(items, scores=items.ids() + count)


def common_pipeline(kind, **options):
    if kind == "rec-builder":
        builder = RecPipelineBuilder(**options)
        builder.scorer(history_scorer)
        return builder.build()
    if kind == "topn":
        return topn_pipeline(history_scorer, **options)
    if kind == "topn-predict":
        return topn_pipeline(history_scorer, predicts_ratings=True, **options)
    if kind == "predict":
        return predict_pipeline(history_scorer, **options)
    return Pipeline.from_config(
        {
            "options": {"base": kind, **options},
            "components": {"scorer": {"code": "tests.pipeline.test_common:history_scorer"}},
        }
    )


PIPELINES = ["rec-builder", "topn", "topn-predict", "predict", "std:topn", "std:topn-predict"]


@fixture
def small_data():
    return from_interactions_df(
        pd.DataFrame(
            {"user_id": [1, 1, 2, 2], "item_id": [11, 12, 13, 14], "rating": [3.0, 4.0, 2.0, 5.0]}
        )
    )


@mark.parametrize("kind", PIPELINES)
def test_common_pipeline_history_lookup_default(kind, small_data):
    pipe = common_pipeline(kind)
    assert isinstance(pipe.component("history-lookup"), UserTrainingHistoryLookup)
    pipe.train(small_data)
    scored = pipe.run("scorer", query=1, items=ItemList(item_ids=[12, 14]))
    np.testing.assert_array_equal(scored.scores(), [14, 16])


@mark.parametrize("kind", PIPELINES)
def test_common_pipeline_without_history_lookup(kind, monkeypatch, small_data):
    pipe = common_pipeline(kind, history_lookup=False)
    assert pipe.node("history-lookup", missing="none") is None

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("disabled history lookup must not be trained")

    monkeypatch.setattr(UserTrainingHistoryLookup, "train", unexpected_lookup)
    pipe.train(small_data)

    history = ItemList(item_ids=[13])
    query = RecQuery(user_id=1, history_items=history)
    items = ItemList(item_ids=[12, 14])
    scored = pipe.run("scorer", query=query, items=items)
    np.testing.assert_array_equal(scored.scores(), [13, 15])
    assert query.history_items is history

    # Reconstructing the concrete configuration must keep lookup disabled.
    restored = Pipeline.from_config(pipe.config.model_dump())
    assert restored.node("history-lookup", missing="none") is None

    if kind != "predict":
        recs = pipe.run("recommender", query=query, n=2)
        np.testing.assert_array_equal(recs.ids(), [14, 12])
        np.testing.assert_array_equal(recs.scores(), [15, 13])
    if kind in ("predict", "topn-predict", "std:topn-predict"):
        preds = pipe.run("rating-predictor", query=query, items=items)
        np.testing.assert_array_equal(preds.scores(), [13, 15])


@mark.parametrize("query", [1, ItemList(item_ids=[13])])
def test_common_pipeline_without_lookup_accepts_query_inputs(query):
    pipe = topn_pipeline(history_scorer, history_lookup=False)
    scored = pipe.run("scorer", query=query, items=ItemList(item_ids=[12, 14]))
    count = 1 if isinstance(query, ItemList) else 0
    np.testing.assert_array_equal(scored.scores(), [12 + count, 14 + count])
