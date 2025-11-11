# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.data import Dataset


def test_item_lists(ml_ds: Dataset):
    ilc = ml_ds.interactions().item_lists()
    assert len(ilc) == ml_ds.user_count
    assert ilc.key_fields == ("user_id",)
