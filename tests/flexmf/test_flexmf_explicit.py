# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import torch
import pandas as pd
from pytest import approx

from lenskit.flexmf import FlexMFExplicitScorer
from lenskit.flexmf._explicit import FlexMFExplicitConfig
from lenskit.testing import BasicComponentTests, ScorerTests
from lenskit.data import ItemList, RecQuery, from_interactions_df


class TestFlexMFExplicitL2(BasicComponentTests, ScorerTests):
    expected_rmse = approx(0.96, abs=0.05)
    component = FlexMFExplicitScorer
    config = FlexMFExplicitConfig(reg_method="L2")


class TestFlexMFExplicitAdam(BasicComponentTests, ScorerTests):
    component = FlexMFExplicitScorer
    config = FlexMFExplicitConfig(reg_method="AdamW")

def test_explicit_pool_query_items():
    ratings = pd.DataFrame({
            "user": [10, 10, 20, 20],
            "item": [1, 2, 2, 3],
            "rating": [4.0, 5.0, 3.0, 2.0]})
    
    data = from_interactions_df(ratings)

    scorer = FlexMFExplicitScorer(epochs=2)
    scorer.train(data)

    scores = scorer(
        RecQuery(history_items=ItemList([1, 2])),
        ItemList([3])).scores()

    assert scores is not None
    device = scorer.model.device

    history_nums = torch.tensor(
        [scorer.items.number(1), scorer.items.number(2)], device=device)
   
    item_num = torch.tensor([scorer.items.number(3)], device=device)

    with torch.no_grad():
        user = scorer.model.i_embed(history_nums).mean(dim=0)
        expected = scorer.model.score_user_vector(user, item_num)
        expected = expected + scorer.global_bias

    assert scores[0] == approx(expected.item())