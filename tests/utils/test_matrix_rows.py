# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for matrix row utilities.
"""

import logging

import numpy as np
import torch

from hypothesis import HealthCheck, given, settings
from pytest import approx

from lenskit.math.sparse import normalize_sparse_rows
from lenskit.testing import sparse_tensors

_log = logging.getLogger(__name__)


@settings(deadline=1000, suppress_health_check=[HealthCheck.too_slow])
@given(sparse_tensors(dtype=np.float64))
def test_sparse_mean_center(tensor: torch.Tensor):
    nr, nc = tensor.shape

    coo = tensor.to_sparse_coo()
    rows = coo.indices()[0, :].numpy()
    counts = np.zeros(nr, dtype=np.int32)
    sums = np.zeros(nr, dtype=np.float64)

    np.add.at(counts, rows, 1)
    np.add.at(sums, rows, coo.values().numpy())
    tgt_means = sums / counts
    tgt_means = np.nan_to_num(tgt_means, nan=0)

    nt, means = normalize_sparse_rows(tensor, "center")
    assert means.shape == torch.Size([nr])

    assert means.numpy() == approx(tgt_means, nan_ok=True, rel=1.0e-6)

    for i in range(nr):
        tr = tensor[i].values().numpy()
        nr = nt[i].values().numpy()
        assert nr == approx(tr - means[i].numpy())


@settings(deadline=1000, suppress_health_check=[HealthCheck.too_slow])
@given(sparse_tensors())
def test_sparse_unit_norm(tensor):
    nr, nc = tensor.shape

    nt, norms = normalize_sparse_rows(tensor, "unit")

    assert norms.numpy() == approx(
        torch.linalg.vector_norm(tensor.to_dense(), dim=1).numpy(), rel=1.0e-3
    )

    for i in range(nr):
        tr = tensor[i].values().numpy()
        nr = nt[i].values().numpy()
        assert nr * norms[i].numpy() == approx(tr)
