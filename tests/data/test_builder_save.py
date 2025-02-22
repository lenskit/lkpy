# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: basic
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from pytest import approx, raises, warns

from lenskit.data import DatasetBuilder
from lenskit.data.dataset import Dataset
from lenskit.data.schema import AllowableTroolean, DataSchema
from lenskit.diagnostics import DataError, DataWarning


def test_save_load_ml(ml_ratings: pd.DataFrame, tmpdir: Path):
    dsb = DatasetBuilder()
    dsb.add_interactions("rating", ml_ratings, entities=["user", "item"], missing="insert")

    ds = dsb.build()

    ds_path = tmpdir / "dataset"
    dsb.save(ds_path)

    assert ds_path.exists()

    schema_path = ds_path / "schema.json"
    assert schema_path.exists()
    schema = DataSchema.model_validate_json(schema_path.read_text("utf8"))
    assert schema == dsb.schema

    assert (ds_path / "item.parquet").exists()
    assert (ds_path / "user.parquet").exists()
    assert (ds_path / "rating.parquet").exists()

    ds2 = Dataset.load(ds_path)

    assert ds2.user_count == ds.user_count
    assert ds2.item_count == ds.item_count
    assert np.all(ds2.users.ids() == ds.users.ids())
    assert np.all(ds2.items.ids() == ds.items.ids())
    assert ds2.interaction_count == ds.interaction_count
