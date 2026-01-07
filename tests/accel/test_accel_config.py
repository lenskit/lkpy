# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit import _accel
from lenskit.parallel import ensure_parallel_init, get_parallel_config


def test_thread_count():
    ensure_parallel_init()
    pc = get_parallel_config()

    tc = _accel.thread_count()
    assert tc == pc.threads
