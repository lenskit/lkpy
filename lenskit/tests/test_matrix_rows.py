# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
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

from lenskit.data.matrix import normalize_sparse_rows, sparse_row_stats
from lenskit.util.test import sparse_tensors

_log = logging.getLogger(__name__)


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(sparse_tensors())
def test_sparse_stats(tensor):
    nr, nc = tensor.shape
    _log.debug("tensor: %d x %d", nr, nc)

    stats = sparse_row_stats(tensor)
    assert stats.means.shape == (nr,)
    assert stats.counts.shape == (nr,)

    assert np.sum(stats.counts.numpy()) == tensor.values().shape[0]

    sums = tensor.sum(dim=1, keepdim=True)
    sums = sums.to_dense().reshape(-1)
    tots = stats.means * stats.counts
    mask = stats.counts.numpy() > 0
    assert tots.numpy()[mask] == approx(sums.numpy()[mask])


@settings(deadline=1000, suppress_health_check=[HealthCheck.too_slow])
@given(sparse_tensors())
def test_sparse_mean_center(tensor):
    nr, nc = tensor.shape

    stats = sparse_row_stats(tensor)
    nt, means = normalize_sparse_rows(tensor, "center")

    assert means.numpy() == approx(stats.means.numpy(), nan_ok=True)

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
