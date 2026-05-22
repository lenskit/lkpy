# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.data import flatten_dict, unflatten_dict


def test_flatten_empty_dict():
    d2 = flatten_dict({})
    assert d2 == {}


def test_unflatten_empty_dict():
    d2 = unflatten_dict({})
    assert d2 == {}


def test_flatten_simple_dict():
    d2 = flatten_dict({"a": 39, "b": "hackem muche"})
    assert d2 == {"a": 39, "b": "hackem muche"}


def test_unflatten_simple_dict():
    d2 = unflatten_dict({"a": 39, "b": "hackem muche"})
    assert d2 == {"a": 39, "b": "hackem muche"}


def test_flatten_nested():
    src = {"foo": 7, "scrolls": {"first": "hackem muche", "second": {"title": "read me"}}}
    out = flatten_dict(src)
    assert out == {"foo": 7, "scrolls.first": "hackem muche", "scrolls.second.title": "read me"}


def test_unflatten_nested():
    src = {"foo": 7, "scrolls.first": "hackem muche", "scrolls.second.title": "read me"}
    out = unflatten_dict(src)
    assert out == {"foo": 7, "scrolls": {"first": "hackem muche", "second": {"title": "read me"}}}
