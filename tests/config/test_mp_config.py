# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.config import ParallelSettings
from lenskit.parallel import effective_cpu_count, is_free_threaded


def test_default_config():
    ps = ParallelSettings()
    ps.resolve_defaults()

    assert ps.num_cpus == effective_cpu_count()
    assert 0 < ps.num_threads <= ps.num_cpus
    assert 0 < ps.num_backend_threads <= ps.num_cpus
    assert ps.num_threads * ps.num_backend_threads <= ps.num_cpus


def test_set_threads():
    ps = ParallelSettings(num_cpus=8, num_threads=4)
    ps.resolve_defaults()

    assert ps.num_cpus == 8
    assert ps.num_threads == 4
    assert ps.num_backend_threads == 2
    if is_free_threaded():
        assert ps.num_batch_jobs == 4
    else:
        assert ps.num_batch_jobs == 1


def test_auto_threads():
    ps = ParallelSettings(num_cpus=8, num_threads=-1)
    ps.resolve_defaults()

    assert ps.num_cpus == 8
    assert ps.num_threads == 8
    assert ps.num_backend_threads == 1
    if is_free_threaded():
        assert ps.num_batch_jobs == 8
    else:
        assert ps.num_batch_jobs == 1


def test_auto_batch():
    ps = ParallelSettings(num_cpus=16, num_batch_jobs=-1)
    ps.resolve_defaults()

    assert ps.num_cpus == 16
    assert ps.num_threads == 8
    assert ps.num_backend_threads == 2
    assert ps.num_batch_jobs == 16
