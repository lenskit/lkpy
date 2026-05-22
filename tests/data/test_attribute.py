# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pandas as pd

from pytest import raises

from lenskit.data import Dataset, DatasetBuilder


def test_attribute_single_value():
    items = pd.DataFrame({"item_id": [42], "name": ["HACKEM MUCHE"]})
    dsb = DatasetBuilder()
    dsb.add_entities("item", items)
    data = dsb.build()
    assert data.item_count == 1

    assert data.entities("item").attribute("name").value() == "HACKEM MUCHE"
    assert data.entities("item").attribute("name").list() == ["HACKEM MUCHE"]


def test_attribute_pydata():
    items = pd.DataFrame({"item_id": [42, 67], "name": ["HACKEM MUCHE", "READ ME"]})
    dsb = DatasetBuilder()
    dsb.add_entities("item", items)
    data = dsb.build()
    assert data.item_count == 2

    assert data.entities("item").attribute("name").list() == ["HACKEM MUCHE", "READ ME"]
    assert data.entities("item").select(ids=[67]).attribute("name").list() == ["READ ME"]
    assert data.entities("item").select(ids=[42]).attribute("name").arrow().to_pylist() == [
        "HACKEM MUCHE"
    ]

    with raises(ValueError):
        assert data.entities("item").attribute("name").value()
    assert data.entities("item").select(ids=[42]).attribute("name").value() == "HACKEM MUCHE"
