# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import itertools
import logging
import multiprocessing as mp

import numpy as np
import torch

from pytest import approx, mark

from lenskit.parallel import invoker
from lenskit.parallel.config import _resolve_parallel_config
from lenskit.testing import set_env_var

_log = logging.getLogger(__name__)


def _mul_op(m, v):
    return m @ v


@mark.slow
@mark.parametrize("pkg,n_jobs", itertools.product(["numpy", "torch"], [None, 1, 2, 4]))
def test_invoke_matrix(pkg, n_jobs, rng: np.random.Generator):
    matrix = rng.normal(size=(1000, 1000))
    vectors = [rng.normal(size=1000) for i in range(100)]
    if pkg == "torch":
        matrix = torch.from_numpy(matrix)
        vectors = [torch.from_numpy(v) for v in vectors]
    with invoker(matrix, _mul_op, n_jobs) as inv:
        mults = inv.map(vectors)
        for rv, v in zip(mults, vectors):
            act_rv = matrix @ v
            assert act_rv == approx(rv, abs=1.0e-6)


def test_proc_count_default():
    with set_env_var("LK_NUM_PROCS", None):
        cfg = _resolve_parallel_config()
        assert cfg.processes == min(mp.cpu_count(), 4)


def test_proc_count_env():
    with set_env_var("LK_NUM_PROCS", "17"):
        assert _resolve_parallel_config().processes == 17
