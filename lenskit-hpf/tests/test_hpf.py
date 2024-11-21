# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import pickle

import numpy as np
import pandas as pd

from pytest import importorskip, mark

from lenskit.algorithms import basic
from lenskit.data import from_interactions_df

hpf = importorskip("lenskit.hpf")

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)


@mark.slow
def test_hpf_train_large(tmp_path, ml_ratings):
    algo = hpf.HPF(20)
    ratings = ml_ratings.assign(rating=ml_ratings.rating + 0.5)
    ds = from_interactions_df(ratings)
    algo.train(ds)

    assert algo.n_users == ratings.user.nunique()
    assert algo.n_items == ratings.item.nunique()

    mfile = tmp_path / "hpf.dat"
    with mfile.open("wb") as mf:
        pickle.dump(algo, mf)

    with mfile.open("rb") as mf:
        a2 = pickle.load(mf)

    assert a2.n_users == algo.n_users
    assert a2.n_items == algo.n_items

    csel = basic.UnratedItemCandidateSelector()
    csel.fit(ds)
    rec = basic.TopN(algo, csel)

    for u in np.random.choice(ratings.user.unique(), size=50, replace=False):
        recs = rec.recommend(u, 50)
        assert len(recs) == 50
        assert recs.item.nunique() == 50


@mark.slow
def test_hpf_train_binary(tmp_path, ml_ratings):
    algo = hpf.HPF(20)
    ratings = ml_ratings.drop(columns=["timestamp", "rating"])
    ds = from_interactions_df(ratings)
    algo.fit(ds)

    assert algo.n_users == ratings.user.nunique()
    assert algo.n_items == ratings.item.nunique()

    mfile = tmp_path / "hpf.dat"
    with mfile.open("wb") as mf:
        pickle.dump(algo, mf)

    with mfile.open("rb") as mf:
        a2 = pickle.load(mf)

    assert a2.n_users == algo.n_users
    assert a2.n_items == algo.n_items

    csel = basic.UnratedItemCandidateSelector()
    csel.fit(ds)
    rec = basic.TopN(algo, csel)

    for u in np.random.choice(ratings.user.unique(), size=50, replace=False):
        recs = rec.recommend(u, 50)
        assert len(recs) == 50
        assert recs.item.nunique() == 50
