# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.knn import AssociationConfig, AssociationScorer
from lenskit.testing import BasicComponentTests, ScorerTests


class TestCondProb(BasicComponentTests, ScorerTests):
    can_score = "some"
    component = AssociationScorer

    expected_ndcg = 0.01


class TestLift(BasicComponentTests, ScorerTests):
    can_score = "some"
    component = AssociationScorer
    config = AssociationConfig(method="lift", damping=10, max_nbrs=1)

    expected_ndcg = 0.01
