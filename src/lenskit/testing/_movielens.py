# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
MovieLens test fixtures and data marks.
"""

import os
from pathlib import Path
from typing import Generator, NamedTuple

import numpy as np
import pandas as pd

import pytest

from lenskit.basic import PopScorer, SoftmaxRanker
from lenskit.batch import recommend
from lenskit.data import Dataset, ItemListCollection, UserIDKey
from lenskit.data.movielens import load_movielens, load_movielens_df
from lenskit.logging import get_logger
from lenskit.pipeline import RecPipelineBuilder
from lenskit.splitting import TTSplit, simple_test_pair

_log = get_logger("lenskit.testing")


ml_test_dir = Path("data/ml-latest-small")
ml_100k_zip = Path("data/ml-100k.zip")
ml_25m_zip = Path("data/ml-25m.zip")

ml_test: Dataset = Dataset(lambda: load_movielens(ml_test_dir))

retrain = os.environ.get("LK_TEST_RETRAIN")


class DemoRecs(NamedTuple):
    split: TTSplit
    recommendations: ItemListCollection[UserIDKey]


@pytest.fixture(scope="session")
def ml_ratings() -> Generator[pd.DataFrame, None, None]:
    """
    Fixture to load the test MovieLens ratings as a data frame. To use this,
    just include it as a parameter in your test::

        def test_thing_with_data(ml_ratings: pd.DataFrame):
            ...

    .. note::
        This is imported in ``conftest.py`` so it is always available in LensKit tests.
    """
    yield load_movielens_df(ml_test_dir)


@pytest.fixture(scope="session")
def ml_ds_unchecked() -> Generator[Dataset, None, None]:
    """
    Fixture to load the MovieLens dataset, without checking for modifications.

    Usually use :func:`ml_ds` instead.
    """
    yield load_movielens(ml_test_dir)


@pytest.fixture(scope="function" if retrain else "module")
def ml_ds(ml_ds_unchecked: Dataset) -> Generator[Dataset, None, None]:
    """
    Fixture to load the MovieLens test dataset.  To use this, just include it as
    a parameter in your test::

        def test_thing_with_data(ml_ds: Dataset):
            ...

    .. note::
        This is imported in ``conftest.py`` so it is always available in LensKit tests.
    """
    log = _log.bind()

    ds = ml_ds_unchecked
    old_rates = ds.interaction_matrix(format="pandas", field="rating", original_ids=True).copy(
        deep=True
    )
    old_ustats = ds.user_stats().copy(deep=True)

    yield ds

    ustats = ds.user_stats()

    rates = ds.interaction_matrix(format="pandas", field="rating", original_ids=True)
    assert rates["rating"].values == pytest.approx(old_rates["rating"].values)

    for col in old_ustats.columns:
        log.info("checking user stats column", column=col)
        assert ustats[col].values == pytest.approx(old_ustats[col].values)


@pytest.fixture
def ml_100k() -> Generator[pd.DataFrame, None, None]:
    """
    Fixture to load the MovieLens 100K dataset (currently as a data frame).  It skips
    the test if the ML100K data is not available.
    """
    if not ml_100k_zip.exists():
        pytest.skip("ML100K data not available")
    yield load_movielens_df(ml_100k_zip)


@pytest.fixture(scope="session")
def ml_20m() -> Generator[Dataset, None, None]:
    if not ml_25m_zip.exists():
        pytest.skip("ML25M data not available")
    yield load_movielens(ml_25m_zip)


@pytest.fixture(scope="session")
def demo_recs() -> DemoRecs:
    """
    A demo set of train, test, and recommendation data.
    """
    rng = np.random.default_rng(42)
    ml_ds = load_movielens(ml_test_dir)
    split = simple_test_pair(ml_ds, f_rates=0.5, rng=rng)

    builder = RecPipelineBuilder()
    builder.scorer(PopScorer())
    builder.ranker(SoftmaxRanker(n=500))
    pipe = builder.build()
    pipe.train(split.train)

    recs = recommend(pipe, list(split.test.keys()), 500, n_jobs=1)
    return DemoRecs(split, recs)


@pytest.fixture(scope="session")
def pop_recs() -> DemoRecs:
    """
    A demo set of train, test, and recommendation data, from most-popular.
    """
    rng = np.random.default_rng(42)
    ml_ds = load_movielens(ml_test_dir)
    split = simple_test_pair(ml_ds, f_rates=0.5, rng=rng)

    builder = RecPipelineBuilder()
    builder.scorer(PopScorer())
    builder.ranker(n=500)
    pipe = builder.build()
    pipe.train(split.train)

    recs = recommend(pipe, list(split.test.keys()), 500, n_jobs=1)
    return DemoRecs(split, recs)
