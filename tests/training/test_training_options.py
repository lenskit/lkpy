# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os

from lenskit.testing import set_env_var
from lenskit.training import TrainingOptions


def test_options_no_env():
    opts = TrainingOptions()
    assert opts.envvar("LK_TEST_VAR") is None


def test_options_local_env():
    opts = TrainingOptions(environment={"LK_TEST_VAR": "HACKEM MUCHE"})
    assert opts.envvar("LK_TEST_VAR") == "HACKEM MUCHE"


def test_options_envvar():
    with set_env_var("LK_TEST_VAR", "FOOBIE BLETCH"):
        opts = TrainingOptions()
        assert opts.envvar("LK_TEST_VAR") == "FOOBIE BLETCH"


def test_options_override_envvar():
    with set_env_var("LK_TEST_VAR", "FOOBIE BLETCH"):
        opts = TrainingOptions(environment={"LK_TEST_VAR": "HACKEM MUCHE"})
        assert opts.envvar("LK_TEST_VAR") == "HACKEM MUCHE"
