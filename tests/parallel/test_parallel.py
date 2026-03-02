# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import itertools
import logging
import multiprocessing as mp

import numpy as np
import torch

from pytest import approx, mark, skip

from lenskit.parallel import invoker
from lenskit.parallel.ray import ray_supported

_log = logging.getLogger(__name__)


def _mul_op(m, v):
    return m @ v


@mark.slow
@mark.parametrize("pkg,n_jobs", itertools.product(["numpy", "torch"], [None, 1, 2, 4, "ray"]))
def test_invoke_matrix(pkg, n_jobs, rng: np.random.Generator):
    if n_jobs == "ray" and not ray_supported():
        skip("ray not supported")

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
