# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests for the Bag of Little Bootstraps implementation.
"""

import sys
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

import pytest
from pytest import approx

from lenskit.diagnostics import DataWarning
from lenskit.stats import BagOfLittleBootstraps


def test_blb_mean():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.0, scale=2.0, size=1000)

    blb = BagOfLittleBootstraps(n_subsamples=5, n_bootstrap=50, rng=42)

    result = blb.analyze(data, np.mean)

    assert result.mean == approx(5.0, abs=0.2)
    assert 0.05 < result.std < 0.5
    assert result.ci_lower < 5.0 < result.ci_upper
    assert len(result.replicates) == 5 * 50


def test_blb_dataframe():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value": rng.normal(loc=5.0, scale=2.0, size=1000),
            "group": rng.choice(["A", "B"], size=1000),
        }
    )

    blb = BagOfLittleBootstraps(rng=42)

    def df_mean(data):
        return data["value"].mean()

    result = blb.analyze(df, df_mean)

    assert result.mean == approx(5.0, abs=0.2)
    assert result.ci_lower < 5.0 < result.ci_upper


def test_blb_str():
    result = BagOfLittleBootstraps(rng=42).analyze(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.mean)
    s = str(result)
    assert "mean=" in s
    assert "±" in s
    assert "—" in s


def test_blb_parallel_no_ray():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    blb = BagOfLittleBootstraps(rng=42)
    with patch.dict(sys.modules, {"ray": None}):
        with warnings.catch_warnings(record=True) as w:
            result = blb.analyze_parallel(data, np.mean)
            assert any(issubclass(warn.category, DataWarning) for warn in w)
            assert any(
                "Ray is not available - falling back to sequential implementation"
                in str(warn.message)
                for warn in w
            )
    assert result.mean == approx(3.0, abs=0.5)


@pytest.mark.skipif(
    pytest.importorskip("ray", reason="Ray not available") is None, reason="Ray not available"
)
def test_blb_parallel():
    import ray

    ray.init(num_cpus=2, ignore_reinit_error=True)

    try:
        rng = np.random.default_rng(42)
        data = rng.normal(loc=5.0, scale=2.0, size=1000)

        blb = BagOfLittleBootstraps(n_subsamples=5, n_bootstrap=50, rng=42)

        result = blb.analyze_parallel(data, np.mean)

        assert result.mean == approx(5.0, abs=0.2)
        assert 0.05 < result.std < 0.5
        assert result.ci_lower < 5.0 < result.ci_upper
        assert len(result.replicates) == 5 * 50

    finally:
        ray.shutdown()


@pytest.mark.skipif(
    pytest.importorskip("ray", reason="Ray not available") is None, reason="Ray not available"
)
def test_blb_parallel_dataframe():
    import ray

    ray.init(num_cpus=2, ignore_reinit_error=True)

    try:
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "value": rng.normal(loc=5.0, scale=2.0, size=1000),
                "group": rng.choice(["A", "B"], size=1000),
            }
        )

        blb = BagOfLittleBootstraps(rng=42)

        def df_mean(data):
            return data["value"].mean()

        result = blb.analyze_parallel(df, df_mean)

        assert result.mean == approx(5.0, abs=0.2)
        assert result.ci_lower < 5.0 < result.ci_upper

    finally:
        ray.shutdown()
