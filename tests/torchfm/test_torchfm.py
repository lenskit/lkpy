# test_torchfm.py
# This file is part of LensKit.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import numpy as np
from pytest import mark

from lenskit.data import from_interactions_df
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests

from lenskit.torchfm import TorchFMScorer, TorchFMConfig

# configure logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

class TestTorchFMScorer(BasicComponentTests, ScorerTests):
    component = TorchFMScorer
    config = TorchFMConfig()

@mark.slow
@mark.parametrize(["embed_dim", "neg_count"], [(8, 1), (16, 2)])
def test_torchfm_train_config(ml_ds, embed_dim, neg_count):
    """
    Test that training runs with various embedding sizes and negative sampling.
    """
    cfg = TorchFMConfig(embed_dim=embed_dim, negative_count=neg_count, epochs=1)
    model = TorchFMScorer(cfg)
    model.train(ml_ds)
    assert hasattr(model, "model_")

@mark.slow
@mark.eval
def test_torchfm_test_accuracy(ml_100k):
    """
    Fit TorchFMScorer on MovieLens-100k and ensure nDCG > 0.
    """
    ds = from_interactions_df(ml_100k)
    cfg = TorchFMConfig(embed_dim=25, epochs=10, batch_size=1024)
    algo = TorchFMScorer(cfg)
    results = quick_measure_model(algo, ds)
    ndcg = results.list_summary().loc["NDCG", "mean"]
    assert ndcg > 0
