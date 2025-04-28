# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Tests on the ML-20M data set.
"""

import logging
from pathlib import Path

import pytest

from lenskit.basic import PopScorer
from lenskit.batch import recommend
from lenskit.data import Dataset, from_interactions_df, load_movielens
from lenskit.pipeline import topn_pipeline

_log = logging.getLogger(__name__)

_ml_path = Path("data/ml-20m.zip")


@pytest.fixture(scope="module")
def ml20m():
    if _ml_path.exists():
        return load_movielens(_ml_path)
    else:
        pytest.skip("ML-20M not available")


@pytest.mark.slow
@pytest.mark.realdata
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_pop_recommend(ml20m: Dataset, rng, n_jobs):
    users = rng.choice(ml20m.users.ids(), 10000, replace=False)
    scorer = PopScorer()
    pipe = topn_pipeline(scorer)

    _log.info("training %s", pipe)
    pipe.train(ml20m)
    _log.info("recommending with %s", pipe)
    recs = recommend(pipe, users, 10, n_jobs=n_jobs)

    assert len(recs) == 10000
