# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

from lenskit.logging._config import _verbose_level


def test_verbose_default():
    assert _verbose_level(0) == logging.INFO


def test_verbose_debug():
    assert _verbose_level(1) == logging.DEBUG


def test_verbose_trace():
    assert _verbose_level(2) == 5


def test_verbose_quiet():
    assert _verbose_level(-1) == logging.WARNING
