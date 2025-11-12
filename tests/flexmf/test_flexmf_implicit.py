# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from itertools import product

from pytest import mark, skip

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
    config = FlexMFImplicitConfig(loss="pairwise", epochs=3)

    def test_skip_retrain(self, ml_ds):
        skip("not needed")

    def test_run_with_doubles(self, ml_ratings):
        skip("FlexMF is fine with doubles")


class TestFlexMFWARP(BasicComponentTests, ScorerTests):
    component = FlexMFImplicitScorer
    config = FlexMFImplicitConfig(loss="warp", epochs=3)

    def test_skip_retrain(self, ml_ds):
        skip("not needed")

    def test_run_with_doubles(self, ml_ratings):
        skip("FlexMF is fine with doubles")


def test_config_defaults():
    cfg = FlexMFImplicitConfig()
    assert cfg.embedding_size == 64


def test_config_exp_ctor():
    cfg = FlexMFImplicitConfig(embedding_size_exp=5)  # type: ignore
    assert cfg.embedding_size == 32


def test_config_exp_dict():
    cfg = FlexMFImplicitConfig.model_validate({"embedding_size_exp": 10})
    assert cfg.embedding_size == 1024


def test_config_exp_json():
    cfg = FlexMFImplicitConfig.model_validate_json('{"embedding_size_exp": 2}')
    assert cfg.embedding_size == 4


@mark.slow
@mark.parametrize(["loss", "reg"], product(["logistic", "pairwise"], ["L2", "AdamW"]))
def test_flexmf_train_config(ml_ds, loss, reg):
    config = FlexMFImplicitConfig(loss=loss, reg_method=reg)
    model = FlexMFImplicitScorer(config)
    print("training", model)
    model.train(ml_ds)

    assert model.model is not None
