# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pandas as pd

from lenskit.data import from_interactions_df
from lenskit.knn.ease import EASEConfig, EASEScorer
from lenskit.logging import get_logger
from lenskit.testing import BasicComponentTests, ScorerTests

_log = get_logger(__name__)

simple_ratings = pd.DataFrame.from_records(
    [
        (1, 6, 4.0),
        (2, 6, 2.0),
        (1, 7, 3.0),
        (2, 7, 2.0),
        (3, 7, 5.0),
        (4, 7, 2.0),
        (1, 8, 3.0),
        (2, 8, 4.0),
        (3, 8, 3.0),
        (4, 8, 2.0),
        (5, 8, 3.0),
        (6, 8, 2.0),
        (1, 9, 3.0),
        (3, 9, 4.0),
    ],
    columns=["user_id", "item_id", "rating"],
)
simple_ds = from_interactions_df(simple_ratings)


class TestItemKNN(BasicComponentTests, ScorerTests):
    can_score = "some"
    component = EASEScorer

    expected_ndcg = 0.01


def test_ease_train():
    algo = EASEScorer()
    algo.train(simple_ds)
