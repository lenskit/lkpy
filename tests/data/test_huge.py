# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Test extremely large datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pytest import mark

from lenskit.data import DatasetBuilder
from lenskit.logging import get_logger

_log = get_logger(__name__)
huge_dir = Path("data/ml-20mx16x32")


@mark.skipif(not huge_dir.exists(), reason="ML 1B not available")
@mark.slow
def test_basic_huge():
    "Test building a 1B data set"

    dsb = DatasetBuilder()
    dsb.add_relationship_class(
        "interaction", ["user", "item"], allow_repeats=False, interaction=True
    )

    n_users = 0
    n_rows = 0
    for file in huge_dir.glob("train*.npz"):
        _log.info("loading segment", file=file.name)
        npz = np.load(file)
        array = next(iter(npz.values()))
        n_users += len(np.unique(array[:, 0]))
        n_rows += array.shape[0]
        dsb.add_interactions(
            "interaction",
            pd.DataFrame(array, columns=["user_id", "item_id"]),
            missing="insert",
        )

    _log.info("building dataset")
    ds = dsb.build()
    assert ds.interaction_count == n_rows
    assert ds.user_count == n_users
