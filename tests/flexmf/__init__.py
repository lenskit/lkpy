# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os

import pytest

pytestmark = pytest.mark.skipif(
    "LLVM_PROFILE_FILE" in os.environ,
    reason="Rust coverage is enabled, but these tests are not worth the cost",
)
