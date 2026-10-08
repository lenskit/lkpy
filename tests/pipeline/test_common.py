# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from pytest import fixture

from lenskit.basic import BiasScorer
from lenskit.basic.history import UserTrainingHistoryLookup
from lenskit.data import ItemList, RecQuery, from_interactions_df
from lenskit.pipeline import Pipeline, RecPipelineBuilder, predict_pipeline, topn_pipeline


@fixture
def small_data():
    return from_interactions_df(
        pd.DataFrame(
            {"user_id": [1, 1, 2, 2], "item_id": [11, 12, 13, 14], "rating": [3.0, 4.0, 2.0, 5.0]}
        )
    )


def check_history_lookup_default(pipe, small_data):
    assert isinstance(pipe.component("history-lookup"), UserTrainingHistoryLookup)
    pipe.train(small_data)
    query = RecQuery(user_id=1)
    scored = pipe.run("scorer", query=query, items=ItemList(item_ids=[12, 14]))
    np.testing.assert_array_equal(scored.scores(), [4.0, 5.0])
    assert query.history_items is not None
    np.testing.assert_array_equal(query.history_items.ids(), [11, 12])
    np.testing.assert_array_equal(query.history_items.field("rating"), [3.0, 4.0])


def check_history_lookup_disabled(pipe, monkeypatch, small_data):
    assert pipe.node("history-lookup", missing="none") is None

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("disabled history lookup must not be trained")

    monkeypatch.setattr(UserTrainingHistoryLookup, "train", unexpected_lookup)
    pipe.train(small_data)

    # The supplied rating is two points above item 13's training rating.
    history = ItemList(item_ids=[13], rating=[4.0])
    query = RecQuery(user_id=1, history_items=history)
    items = ItemList(item_ids=[12, 14])
    scored = pipe.run("scorer", query=query, items=items)
    np.testing.assert_array_equal(scored.scores(), [6.0, 7.0])
    assert query.history_items is history

    # Reconstructing the concrete configuration must keep lookup disabled.
    restored = Pipeline.from_config(pipe.config.model_dump())
    assert restored.node("history-lookup", missing="none") is None
    return query


def check_recommendations(pipe, query):
    candidates = pipe.run("candidate-selector", query=query)
    np.testing.assert_array_equal(candidates.ids(), [11, 12, 14])
    recs = pipe.run("recommender", query=query, n=2)
    np.testing.assert_array_equal(recs.ids(), [14, 12])
    np.testing.assert_array_equal(recs.scores(), [7.0, 6.0])


def check_rating_predictions(pipe, query):
    items = ItemList(item_ids=[12, 14])
    preds = pipe.run("rating-predictor", query=query, items=items)
    np.testing.assert_array_equal(preds.scores(), [6.0, 7.0])
    fallback = pipe.run("fallback-predictor", query=query, items=items)
    np.testing.assert_array_equal(fallback.scores(), [6.0, 7.0])


def test_rec_builder_history_lookup_default(small_data):
    builder = RecPipelineBuilder()
    builder.scorer(BiasScorer())
    check_history_lookup_default(builder.build(), small_data)


def test_topn_pipeline_history_lookup_default(small_data):
    pipe = topn_pipeline(BiasScorer())
    check_history_lookup_default(pipe, small_data)


def test_topn_predict_pipeline_history_lookup_default(small_data):
    pipe = topn_pipeline(BiasScorer(), predicts_ratings=True)
    check_history_lookup_default(pipe, small_data)


def test_predict_pipeline_history_lookup_default(small_data):
    pipe = predict_pipeline(BiasScorer())
    check_history_lookup_default(pipe, small_data)


def test_topn_config_history_lookup_default(small_data):
    pipe = Pipeline.from_config(
        {
            "options": {"base": "std:topn"},
            "components": {"scorer": {"class": "lenskit.basic.BiasScorer"}},
        }
    )
    check_history_lookup_default(pipe, small_data)


def test_topn_predict_config_history_lookup_default(small_data):
    pipe = Pipeline.from_config(
        {
            "options": {"base": "std:topn-predict"},
            "components": {"scorer": {"class": "lenskit.basic.BiasScorer"}},
        }
    )
    check_history_lookup_default(pipe, small_data)


def test_rec_builder_without_history_lookup(monkeypatch, small_data):
    builder = RecPipelineBuilder(history_lookup=False)
    builder.scorer(BiasScorer())
    pipe = builder.build()
    query = check_history_lookup_disabled(pipe, monkeypatch, small_data)
    check_recommendations(pipe, query)


def test_topn_pipeline_without_history_lookup(monkeypatch, small_data):
    pipe = topn_pipeline(BiasScorer(), history_lookup=False)
    query = check_history_lookup_disabled(pipe, monkeypatch, small_data)
    check_recommendations(pipe, query)


def test_topn_predict_pipeline_without_history_lookup(monkeypatch, small_data):
    pipe = topn_pipeline(BiasScorer(), predicts_ratings=True, history_lookup=False)
    query = check_history_lookup_disabled(pipe, monkeypatch, small_data)
    check_recommendations(pipe, query)
    check_rating_predictions(pipe, query)


def test_predict_pipeline_without_history_lookup(monkeypatch, small_data):
    pipe = predict_pipeline(BiasScorer(), history_lookup=False)
    query = check_history_lookup_disabled(pipe, monkeypatch, small_data)
    check_rating_predictions(pipe, query)


def test_topn_config_without_history_lookup(monkeypatch, small_data):
    pipe = Pipeline.from_config(
        {
            "options": {"base": "std:topn", "history_lookup": False},
            "components": {"scorer": {"class": "lenskit.basic.BiasScorer"}},
        }
    )
    query = check_history_lookup_disabled(pipe, monkeypatch, small_data)
    check_recommendations(pipe, query)


def test_topn_predict_config_without_history_lookup(monkeypatch, small_data):
    pipe = Pipeline.from_config(
        {
            "options": {"base": "std:topn-predict", "history_lookup": False},
            "components": {"scorer": {"class": "lenskit.basic.BiasScorer"}},
        }
    )
    query = check_history_lookup_disabled(pipe, monkeypatch, small_data)
    check_recommendations(pipe, query)
    check_rating_predictions(pipe, query)


def test_topn_without_history_lookup_accepts_user_id(small_data):
    pipe = topn_pipeline(BiasScorer(), history_lookup=False)
    pipe.train(small_data)
    scored = pipe.run("scorer", query=1, items=ItemList(item_ids=[12, 14]))
    np.testing.assert_array_equal(scored.scores(), [4.0, 5.0])
    candidates = pipe.run("candidate-selector", query=1)
    np.testing.assert_array_equal(candidates.ids(), [11, 12, 13, 14])


def test_topn_without_history_lookup_accepts_item_list(small_data):
    pipe = topn_pipeline(BiasScorer(), history_lookup=False)
    pipe.train(small_data)
    query = ItemList(item_ids=[13], rating=[4.0])
    scored = pipe.run("scorer", query=query, items=ItemList(item_ids=[12, 14]))
    np.testing.assert_array_equal(scored.scores(), [6.0, 7.0])
    candidates = pipe.run("candidate-selector", query=query)
    np.testing.assert_array_equal(candidates.ids(), [11, 12, 14])
