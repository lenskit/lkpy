# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:component.*is local:lenskit.diagnostics.PipelineWarning"
)
