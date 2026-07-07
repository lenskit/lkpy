# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pydantic import ValidationError

from pytest import raises

from lenskit.tuning.spec import SearchParam


def test_require_range():
    with raises(ValidationError):
        SearchParam(type="int")
    with raises(ValidationError):
        SearchParam(type="int", min=0)
    with raises(ValidationError):
        SearchParam(type="int", max=0)


def test_bool_accepts_no_range():
    spec = SearchParam(type="bool")
    assert spec.type == "bool"


def test_pow2_must_be_int():
    with raises(ValidationError):
        SearchParam(type="float", min=2, max=10, scale="pow2")


def test_require_choices():
    with raises(ValidationError):
        SearchParam(type="choice")
