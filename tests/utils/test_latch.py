# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.util import Latch


def test_initial_latch():
    x = Latch()

    assert not x.is_latched()


def test_latch():
    x = Latch()

    assert not x.is_latched()
    assert x.latch()
    assert x.is_latched()


def test_relatch():
    x = Latch()

    assert not x.is_latched()
    assert x.latch()
    assert not x.latch()
    assert x.is_latched()


def test_reset():
    x = Latch()

    assert not x.is_latched()
    assert x.latch()
    assert x.is_latched()

    x.reset()
    assert not x.is_latched()
