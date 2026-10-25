# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pathlib import Path
from shutil import copytree

import numpy as np

from pytest import mark, raises

from lenskit.data import Dataset, load_movielens, load_movielens_df, load_ms_web
from lenskit.testing import ml_test_dir

MSWEB_TRAIN = Path("data/anonymous-msweb.data.gz")
MSWEB_TEST = Path("data/anonymous-msweb.test.gz")


@mark.skipif(not MSWEB_TRAIN.exists(), reason="msweb data not present")
@mark.realdata
def test_msweb_train():
    ds = load_ms_web(MSWEB_TRAIN)
    assert 32000 < ds.entities("session").count() < 33000
    assert 250 < ds.item_count < 300


@mark.skipif(not MSWEB_TEST.exists(), reason="msweb data not present")
@mark.realdata
def test_msweb_test():
    ds = load_ms_web(MSWEB_TEST, "collection")
    assert len(ds) >= 500
