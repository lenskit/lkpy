# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pydantic import ValidationError

from pytest import raises

from lenskit.schemas.tuning import SearchConfig, SearchParam


def test_merge_configs():
    first = SearchConfig(max_epochs=42)
    second = SearchConfig(min_epochs=17)
    both = first.merge(second)
    assert both.max_epochs == 42
    assert both.min_epochs == 17


def test_require_range():
    with raises(TypeError):
        SearchParam(type="int")
    with raises(TypeError):
        SearchParam(type="int", min=0)
    with raises(TypeError):
        SearchParam(type="int", max=0)


def test_bool_accepts_no_range():
    spec = SearchParam(type="bool")
    assert spec.type == "bool"


def test_pow2_must_be_int():
    with raises(ValueError):
        SearchParam(type="float", min=2, max=10, scale="pow2")


def test_require_choices():
    with raises(ValueError):
        SearchParam(type="choice")
