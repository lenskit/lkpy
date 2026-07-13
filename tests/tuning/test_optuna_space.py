# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import math

from pytest import importorskip

from lenskit.schemas.tuning import SearchParam

optuna = importorskip("optuna")


NTRIALS = 1000


def test_log2_space():
    from lenskit.tuning._optuna import _ask_space

    space = {"n": SearchParam(type="int", scale="pow2", min=8, max=1024)}
    study = optuna.create_study()

    for _t in range(NTRIALS):
        trial = study.ask()
        for p in range(10):
            pt = _ask_space(trial, space)
            x = pt["n"]
            assert isinstance(x, int)
            assert x >= 8
            assert x <= 1024
            base = math.log2(x)
            assert int(base) == base
