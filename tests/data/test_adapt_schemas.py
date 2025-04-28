# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from pytest import raises, warns

from lenskit.data.adapt import normalize_columns
from lenskit.data.types import AliasedColumn

SCROLLS = pd.DataFrame(
    {"appearance": ["FOOBIE BLETCH", "HACKEM MUCHE"], "type": ["identify", "teleport"]}
)


def test_normalize_column_name():
    df = normalize_columns(SCROLLS, "appearance")
    assert np.all(df.columns == ["appearance", "type"])


def test_normalize_column_from_index():
    df = normalize_columns(SCROLLS.set_index("appearance"), "appearance")
    assert np.all(df.columns == ["appearance", "type"])


def test_normalize_missing_column():
    with raises(KeyError):
        normalize_columns(SCROLLS, "bob")


def test_normalize_alias():
    col = AliasedColumn("random", ["appearance"])
    df = normalize_columns(SCROLLS, col)
    assert np.all(df.columns == ["random", "type"])


def test_normalize_alias_index():
    col = AliasedColumn("random", ["appearance"])
    df = normalize_columns(SCROLLS.set_index("appearance"), col)
    assert np.all(df.columns == ["random", "type"])


def test_normalize_missing_alias():
    col = AliasedColumn("random", ["color"])
    with raises(KeyError):
        normalize_columns(SCROLLS, col)


def test_normalize_alias_warn():
    col = AliasedColumn("random", ["appearance"], warn=True)
    with warns(DeprecationWarning):
        df = normalize_columns(SCROLLS, col)
    assert np.all(df.columns == ["random", "type"])


def test_normalize_alias_index_warn():
    col = AliasedColumn("random", ["appearance"], warn=True)
    with warns(DeprecationWarning):
        df = normalize_columns(SCROLLS.set_index("appearance"), col)
    assert np.all(df.columns == ["random", "type"])
