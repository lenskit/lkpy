# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import lenskit.util.test as lktu
from lenskit.algorithms.basic import PopScore
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.ranking import PlackettLuce
from lenskit.util.test import ml_ds, ml_ratings  # noqa: F401


def test_plackett_luce_rec(ml_ds):
    pop = PopScore()
    algo = PlackettLuce(pop, rng_spec=(1005, "user"))
    algo.fit(ml_ds)

    items = ml_ds.items.ids()
    nitems = len(items)

    recs1 = algo.recommend(2038, 100)
    recs2 = algo.recommend(2028, 100)
    assert len(recs1) == 100
    assert len(recs2) == 100

    # we don't get exactly the same set of recs
    assert set(recs1["item"]) != set(recs2["item"])

    recs_all = algo.recommend(2038)
    assert len(recs_all) == nitems
    assert set(items) == set(recs_all["item"])


def test_plackett_luce_pred(ml_ds):
    bias = Bias()
    algo = PlackettLuce(bias, rng_spec="user")
    algo.fit(ml_ds)

    items = ml_ds.items.ids()
    nitems = len(items)

    recs1 = algo.recommend(2038, 100)
    recs2 = algo.recommend(2028, 100)
    assert len(recs1) == 100
    assert len(recs2) == 100

    # we don't get exactly the same set of recs
    assert set(recs1["item"]) != set(recs2["item"])

    recs_all = algo.recommend(2038)
    assert len(recs_all) == nitems
    assert set(items) == set(recs_all["item"])
