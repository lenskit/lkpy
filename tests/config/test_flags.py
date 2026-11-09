# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from pytest import raises

from lenskit.config import parse_flag


def test_parse_none():
    assert parse_flag(None) is False
    assert parse_flag(None, default=True) is True


def test_parse_bool():
    assert parse_flag(True) is True


def test_parse_false():
    assert parse_flag(False) is False
    assert parse_flag(False) is False


def test_parse_int():
    assert parse_flag(0) is False
    assert parse_flag(42) is True


def test_parse_str_bool():
    assert parse_flag("true") is True
    assert parse_flag("false") is False


def test_parse_str_yn():
    assert parse_flag("yes") is True
    assert parse_flag("Y") is True
    assert parse_flag("no") is False
    assert parse_flag("N") is False
    assert parse_flag("N") is False


def test_parse_invalid_str():
    with raises(ValueError, match=r"unrecognized boolean.*ACTIVE=fish"):
        parse_flag("fish", name="ACTIVE")

    with raises(ValueError, match=r"unrecognized boolean.*fish"):
        parse_flag("fish")
