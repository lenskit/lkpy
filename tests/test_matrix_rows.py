"""
Tests for matrix row utilities.
"""

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import HealthCheck, assume, given, settings
from pytest import approx, mark

from lenskit.data.matrix import sparse_row_stats
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
