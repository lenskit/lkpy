# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pytest import approx

from lenskit.flexmf import FlexMFExplicitScorer
from lenskit.flexmf._explicit import FlexMFExplicitConfig
from lenskit.testing import BasicComponentTests, ScorerTests


class TestFlexMFExplicitL2(BasicComponentTests, ScorerTests):
    expected_rmse = approx(0.96, abs=0.05)
    component = FlexMFExplicitScorer
    config = FlexMFExplicitConfig(reg_method="L2")


class TestFlexMFExplicitAdam(BasicComponentTests, ScorerTests):
    component = FlexMFExplicitScorer
    config = FlexMFExplicitConfig(reg_method="AdamW")
