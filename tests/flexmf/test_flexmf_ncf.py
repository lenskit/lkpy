# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from itertools import product

import torch

from pytest import mark, skip

from lenskit.flexmf._ncf import FlexMFNCFConfig, FlexMFNCFModel, FlexMFNCFScorer
from lenskit.testing import BasicComponentTests, ScorerTests


class TestFlexMFNCF(BasicComponentTests, ScorerTests):
    expected_ndcg = (0.01, 0.25)
    component = FlexMFNCFScorer
    config = FlexMFNCFConfig(
        epochs=3, embedding_size=16, mlp_embedding_size=16, mlp_layers=[32, 16, 8]
    )

    def test_skip_retrain(self, ml_ds):
        skip("not needed")

    def test_run_with_doubles(self, ml_ratings):
        skip("FlexMF is fine with doubles")


def test_ncf_config_defaults():
    cfg = FlexMFNCFConfig()
    assert cfg.embedding_size == 64  # inherited default from FlexMFConfigBase
    assert cfg.mlp_embedding_size == 8
    assert cfg.mlp_layers == [16, 8, 4]
    assert cfg.user_bias is False
    assert cfg.item_bias is False


@mark.parametrize(["user_bias", "item_bias"], product([True, False], [True, False]))
def test_ncf_biases(ml_ds, user_bias, item_bias):
    config = FlexMFNCFConfig(epochs=1, user_bias=user_bias, item_bias=item_bias)
    model = FlexMFNCFScorer(config)
    model.train(ml_ds)

    ncf = model.model
    assert isinstance(ncf, FlexMFNCFModel)
    gmf = ncf.gmf_model
    assert (gmf.u_bias is not None) == user_bias
    assert (gmf.i_bias is not None) == item_bias

    users = torch.tensor([0])
    items = torch.arange(5)
    scores = ncf(users, items)
    assert scores.shape == (5,)
    assert torch.all(torch.isfinite(scores))

    # biases must actually contribute to the score
    if gmf.i_bias is not None:
        with torch.no_grad():
            gmf.i_bias.weight[items] += 1.0
            assert torch.allclose(ncf(users, items), scores + 1.0, atol=1e-5)


def test_ncf_config_negative_default():
    cfg = FlexMFNCFConfig(loss="pairwise")
    assert cfg.loss == "pairwise"
    assert cfg.selected_negative_strategy() == "uniform"


@mark.slow
@mark.parametrize(["loss", "reg"], product(["logistic", "pairwise"], ["AdamW"]))
def test_flexmf_ncf_train_config(ml_ds, loss, reg):
    config = FlexMFNCFConfig(loss=loss, reg_method=reg, epochs=1)
    model = FlexMFNCFScorer(config)
    print("training", model)
    model.train(ml_ds)

    assert model.model is not None
