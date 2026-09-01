# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from itertools import product

import numpy as np
import torch

from pytest import mark, skip

from lenskit.data import ItemList, RecQuery
from lenskit.flexmf import FlexMFImplicitConfig, FlexMFImplicitScorer
from lenskit.testing import BasicComponentTests, ScorerTests


class TestFlexMFImplicit(BasicComponentTests, ScorerTests):
    expected_ndcg = (0.01, 0.25)
    component = FlexMFImplicitScorer
    config = FlexMFImplicitConfig(epochs=3)

    def test_skip_retrain(self, ml_ds):
        skip("not needed")

    def test_run_with_doubles(self, ml_ratings):
        skip("FlexMF is fine with doubles")


class TestFlexMFBPR(BasicComponentTests, ScorerTests):
    expected_ndcg = (0.01, 0.25)
    component = FlexMFImplicitScorer
    config = FlexMFImplicitConfig(preset="bpr", epochs=3)

    def test_skip_retrain(self, ml_ds):
        skip("not needed")

    def test_run_with_doubles(self, ml_ratings):
        skip("FlexMF is fine with doubles")


class TestFlexMFWARP(BasicComponentTests, ScorerTests):
    component = FlexMFImplicitScorer
    config = FlexMFImplicitConfig(preset="warp", epochs=3)

    def test_skip_retrain(self, ml_ds):
        skip("not needed")

    def test_run_with_doubles(self, ml_ratings):
        skip("FlexMF is fine with doubles")


class TestFlexMFGCN(BasicComponentTests, ScorerTests):
    expected_ndcg = (0.01, 0.25)
    component = FlexMFImplicitScorer
    config = FlexMFImplicitConfig(preset="lightgcn", epochs=3)

    def test_skip_retrain(self, ml_ds):
        skip("not needed")

    def test_run_with_doubles(self, ml_ratings):
        skip("FlexMF is fine with doubles")


def test_config_defaults():
    cfg = FlexMFImplicitConfig()
    assert cfg.embedding_size == 64
    assert cfg.user_embeddings is True


def test_config_exp_ctor():
    cfg = FlexMFImplicitConfig(embedding_size_exp=5)  # type: ignore
    assert cfg.embedding_size == 32


def test_config_exp_dict():
    cfg = FlexMFImplicitConfig.model_validate({"embedding_size_exp": 10})
    assert cfg.embedding_size == 1024


def test_config_exp_json():
    cfg = FlexMFImplicitConfig.model_validate_json('{"embedding_size_exp": 2}')
    assert cfg.embedding_size == 4


def test_config_negative_default():
    cfg = FlexMFImplicitConfig(loss="pairwise")
    assert cfg.loss == "pairwise"
    assert cfg.selected_negative_strategy() == "uniform"


def test_config_negative_default_warp():
    cfg = FlexMFImplicitConfig(loss="warp")
    assert cfg.loss == "warp"
    assert cfg.selected_negative_strategy() == "misranked"


def test_config_preset():
    cfg = FlexMFImplicitConfig(preset="warp")
    assert cfg.loss == "warp"
    assert cfg.negative_strategy == "misranked"
    assert cfg.user_embeddings == "prefer"


@mark.parametrize("preset", ["bpr", "warp", "lightgcn"])
def test_presets_prefer_user_embeddings(preset):
    cfg = FlexMFImplicitConfig(preset=preset)
    assert cfg.user_embeddings == "prefer"


@mark.slow
@mark.parametrize(["loss", "reg"], product(["logistic", "pairwise"], ["L2", "AdamW"]))
def test_flexmf_train_config(ml_ds, loss, reg):
    config = FlexMFImplicitConfig(loss=loss, reg_method=reg)
    model = FlexMFImplicitScorer(config)
    print("training", model)
    model.train(ml_ds)

    assert model.model is not None


@mark.parametrize("user_id", [None, "not-a-trained-user"])
def test_flexmf_pool_query_items_for_unknown_user(ml_ds, user_id):
    config = FlexMFImplicitConfig(preset="bpr", epochs=1, embedding_size=8)
    scorer = FlexMFImplicitScorer(config)
    scorer.train(ml_ds)

    item_ids = list(ml_ds.items.ids())
    history = ItemList(item_ids[:2])
    candidates = ItemList(item_ids[2:5])

    query = RecQuery(user_id=user_id, history_items=history)
    result = scorer(query, candidates)

    scores = result.scores()
    assert scores is not None
    assert np.all(np.isfinite(np.asarray(scores)))


def test_flexmf_pool_is_mean_of_item_embeddings(ml_ds):
    config = FlexMFImplicitConfig(preset="bpr", epochs=3, embedding_size=8, user_embeddings=True)

    scorer = FlexMFImplicitScorer(config)
    scorer.train(ml_ds)

    item_ids = list(ml_ds.items.ids())
    history = ItemList(item_ids[:2])
    candidates = ItemList(item_ids[2:5])

    scores = scorer(RecQuery(history_items=history), candidates).scores()

    assert scores is not None

    device = scorer.model.device

    history_nums = history.numbers(vocabulary=scorer.items, missing="negative", format="torch").to(
        device
    )

    candidate_nums = candidates.numbers(
        vocabulary=scorer.items, missing="negative", format="torch"
    ).to(device)

    expected_user = scorer.model.i_embed(history_nums).mean(dim=0)
    expected_score = scorer.model.score_user_vector(expected_user, candidate_nums)

    actual_tensor = torch.as_tensor(scores, device=device)

    torch.testing.assert_close(actual_tensor, expected_score)


def test_flexmf_known_user_uses_pool_by_default(ml_ds):
    config = FlexMFImplicitConfig(preset="bpr", epochs=3, embedding_size=8, user_embeddings=True)

    scorer = FlexMFImplicitScorer(config)
    scorer.train(ml_ds)

    user_id = next(iter(ml_ds.users.ids()))
    item_ids = list(ml_ds.items.ids())

    history = ItemList(item_ids[:2])
    candidates = ItemList(item_ids[2:5])

    known_query = RecQuery(user_id=user_id, history_items=history)
    anonymous_query = RecQuery(history_items=history)

    known_scores = scorer(known_query, candidates).scores()
    anonymous_scores = scorer(anonymous_query, candidates).scores()

    assert known_scores is not None
    assert anonymous_scores is not None

    np.testing.assert_allclose(known_scores, anonymous_scores)


def test_flexmf_prefer_uses_trained_embedding_for_known_user(ml_ds):
    config = FlexMFImplicitConfig(preset="bpr", epochs=3, embedding_size=8)

    scorer = FlexMFImplicitScorer(config)
    scorer.train(ml_ds)

    user_id = next(iter(ml_ds.users.ids()))
    item_ids = list(ml_ds.items.ids())

    history = ItemList(item_ids[:2])
    candidates = ItemList(item_ids[2:5])

    normal_scores = scorer(user_id, candidates).scores()

    history_scores = scorer(
        RecQuery(user_id=user_id, history_items=history),
        candidates,
    ).scores()

    assert normal_scores is not None
    assert history_scores is not None

    np.testing.assert_allclose(normal_scores, history_scores)


def test_flexmf_pool_ignores_unknown_history_items(ml_ds):
    config = FlexMFImplicitConfig(preset="bpr", epochs=3, embedding_size=8, user_embeddings=True)

    scorer = FlexMFImplicitScorer(config)
    scorer.train(ml_ds)

    item_ids = list(ml_ds.items.ids())
    known_item = item_ids[0]
    candidates = ItemList(item_ids[1:4])

    with_unknown = scorer(
        RecQuery(history_items=ItemList([known_item, max(item_ids) + 1])), candidates
    ).scores()

    known_only = scorer(RecQuery(history_items=ItemList([known_item])), candidates).scores()

    assert with_unknown is not None
    assert known_only is not None

    np.testing.assert_allclose(with_unknown, known_only)


def test_flexmf_unknown_user_with_no_known_history_items_is_unscorable(ml_ds):
    config = FlexMFImplicitConfig(preset="bpr", epochs=3, embedding_size=8)

    scorer = FlexMFImplicitScorer(config)
    scorer.train(ml_ds)

    item_ids = list(ml_ds.items.ids())
    candidates = ItemList(item_ids[:3])

    query = RecQuery(user_id="not-a-trained-user", history_items=ItemList(["not-a-trained-item"]))

    scores = scorer(query, candidates).scores()

    assert scores is not None
    assert np.all(np.isnan(scores))


def test_flexmf_known_user_falls_back_when_pooling_fails(ml_ds):
    config = FlexMFImplicitConfig(preset="bpr", epochs=3, embedding_size=8, user_embeddings=True)

    scorer = FlexMFImplicitScorer(config)
    scorer.train(ml_ds)

    user_id = next(iter(ml_ds.users.ids()))
    item_ids = list(ml_ds.items.ids())
    candidates = ItemList(item_ids[:3])

    regular_scores = scorer(user_id, candidates).scores()

    query_scores = scorer(
        RecQuery(user_id=user_id, history_items=ItemList(["not-a-trained-item"])), candidates
    ).scores()

    assert regular_scores is not None
    assert query_scores is not None

    np.testing.assert_allclose(query_scores, regular_scores)


def test_flexmf_disable_trained_user_embeddings(ml_ds):
    config = FlexMFImplicitConfig(epochs=3, embedding_size=8, user_embeddings=False)

    scorer = FlexMFImplicitScorer(config)
    scorer.train(ml_ds)

    user_id = next(iter(ml_ds.users.ids()))
    item_ids = list(ml_ds.items.ids())
    candidates = ItemList(item_ids[:3])

    scores = scorer(user_id, candidates).scores()

    assert scores is not None
    assert np.all(np.isnan(scores))


def test_flexmf_disable_trained_embeddings_still_pools(ml_ds):
    config = FlexMFImplicitConfig(epochs=3, embedding_size=8, user_embeddings=False)

    scorer = FlexMFImplicitScorer(config)
    scorer.train(ml_ds)

    item_ids = list(ml_ds.items.ids())
    history = ItemList(item_ids[:2])
    candidates = ItemList(item_ids[2:5])

    scores = scorer(RecQuery(history_items=history), candidates).scores()

    assert scores is not None
    assert np.all(np.isfinite(scores))
