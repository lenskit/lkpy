# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from itertools import product

from pytest import mark, skip

from lenskit.flexmf._ncf import FlexMFNCFConfig, FlexMFNCFScorer
from lenskit.testing import BasicComponentTests, ScorerTests


class TestFlexMFNCF(BasicComponentTests, ScorerTests):
    expected_ndcg = (0.01, 0.25)
    component = FlexMFNCFScorer
    config = FlexMFNCFConfig(epochs=3, gmf_embedding_size=16, mlp_embedding_size=16, mlp_layers=[32, 16, 8])

    def test_skip_retrain(self, ml_ds):
        skip("not needed")

    def test_run_with_doubles(self, ml_ratings):
        skip("FlexMF is fine with doubles")


def test_ncf_config_defaults():
    cfg = FlexMFNCFConfig()
    assert cfg.gmf_embedding_size == 8
    assert cfg.mlp_embedding_size == 8
    assert cfg.mlp_layers == [16, 8, 4]


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
