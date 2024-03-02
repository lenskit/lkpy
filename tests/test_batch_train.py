# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.batch import train_isolated
from lenskit.util.test import ml_test
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import TopN
from lenskit.algorithms.bias import Bias


def test_train_isolate():
    algo = Bias()
    algo = Recommender.adapt(algo)

    saved = train_isolated(algo, ml_test.ratings)
    try:
        trained = saved.get()
        assert isinstance(trained, TopN)
        recs = trained.recommend(10, 10)
        assert len(recs) == 10
        del recs, trained
    finally:
        saved.close()


def test_train_isolate_file(tmp_path):
    fn = tmp_path / "saved.bpk"
    algo = Bias()
    algo = Recommender.adapt(algo)

    saved = train_isolated(algo, ml_test.ratings, file=fn)
    try:
        assert saved.path == fn
        trained = saved.get()
        assert isinstance(trained, TopN)
        recs = trained.recommend(10, 10)
        assert len(recs) == 10
        del recs, trained
    finally:
        saved.close()
