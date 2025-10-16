# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit._accel import slim as _slim_accel
from lenskit.data import Dataset


def test_slim_trainer(ml_ds: Dataset):
    "Test internal SLIM training function."
    _matrix = ml_ds.interactions().matrix().csr_structure(format="arrow")
