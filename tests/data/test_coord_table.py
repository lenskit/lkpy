# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit._accel import data


def test_empty():
    tbl = data.CoordinateTable(5)
    assert len(tbl) == 0
    assert tbl.dimensions() == 5
    assert tbl.unique_count() == 0
