# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

import hypothesis.strategies as st
from hypothesis import given

from lenskit.tuning._stopping import PlateauStopRule


def test_empty():
    rule = PlateauStopRule(mode="max")

    assert not rule.should_stop([])


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=4))
def test_grace_always_passes(xs):
    rule = PlateauStopRule(mode="max")

    assert not rule.should_stop(xs)


def test_linear_keep_going():
    rule = PlateauStopRule(mode="max")
    vals = np.linspace(0.2, 1.0, 25)
    assert not rule.should_stop(vals)


def test_stall_stop():
    rule = PlateauStopRule(mode="max", check_iters=2)
    vals = np.concatenate([np.linspace(0.2, 1.0, 25), [1.0, 1.0]])
    assert rule.should_stop(vals)


def test_stall_stop_min():
    rule = PlateauStopRule(mode="min", check_iters=2)
    vals = np.concatenate([np.linspace(1.0, 0.2, 25), [0.19999, 0.19998]])
    assert rule.should_stop(vals)
