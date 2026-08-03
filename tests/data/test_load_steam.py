# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from pytest import mark

from lenskit.data import Vocabulary
from lenskit.data.sources.steam import load_steam

STEAM_DIR = Path("data/steam")
AU_FILE = STEAM_DIR / "australian_users_items.json.gz"


@mark.skipif(not AU_FILE.exists(), reason="input data does not exist")
@mark.realdata
def test_steam_australia():
    data = load_steam(AU_FILE)

    # do we have the right number of entities?
    assert 87000 < data.user_count < 89000
    assert 10000 < data.item_count < 12000

    # do we about the right number of interactions?
    ints = data.interactions()
    assert ints.count() >= 1_000_000
