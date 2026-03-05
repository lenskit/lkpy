# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os

import pytest

if "LLVM_PROFILE_FILE" in os.environ:
    pytest.skip(
        "Rust coverage is enabled, but these tests are not worth the cost", allow_module_level=True
    )
