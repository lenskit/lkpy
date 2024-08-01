# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from typing import assert_type
from uuid import UUID

from pytest import raises

from lenskit.pipeline import Node, Pipeline


def test_init_empty():
    pipe = Pipeline()
    assert len(pipe.nodes) == 0


def test_create_input():
    "create an input node"
    pipe = Pipeline()
    src = pipe.create_input("user", int, str)
    assert_type(src, Node[int | str])
    assert isinstance(src, Node)
    assert src.name == "user"
    assert src.types == set([int, str])

    assert len(pipe.nodes) == 1
    assert pipe.node("user") is src


def test_dup_input_fails():
    "create an input node"
    pipe = Pipeline()
    pipe.create_input("user", int, str)

    with raises(ValueError, match="has node"):
        pipe.create_input("user", UUID)
