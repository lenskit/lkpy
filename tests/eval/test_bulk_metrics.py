# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pandas as pd  # noqa: 401

from pytest import approx

from lenskit.data import ItemListCollection
from lenskit.data.adapt import ITEM_COMPAT_COLUMN, USER_COMPAT_COLUMN
from lenskit.metrics.basic import ListLength
from lenskit.metrics.bulk import RunAnalysis
from lenskit.metrics.predict import RMSE
from lenskit.metrics.ranking import DCG, NDCG, RBP, Precision, RecipRank
from lenskit.testing import demo_recs, ml_ratings


def test_bulk_measure_function(ml_ratings: pd.DataFrame):
    bms = RunAnalysis()
    bms.add_metric(ListLength(), "length")
    bms.add_metric(RMSE)

    data = ItemListCollection.from_df(
        ml_ratings.rename(columns={"rating": "score"}), USER_COMPAT_COLUMN, ITEM_COMPAT_COLUMN
    )
    truth = ItemListCollection.from_df(ml_ratings, USER_COMPAT_COLUMN, ITEM_COMPAT_COLUMN)

    metrics = bms.measure(data, truth)
    stats = metrics.list_summary()
    assert stats.loc["length", "mean"] == approx(ml_ratings["user_id"].value_counts().mean())
    assert stats.loc["RMSE", "mean"] == approx(0)


def test_recs(demo_recs):
    split, recs = demo_recs

    bms = RunAnalysis()
    bms.add_metric(ListLength())
    bms.add_metric(Precision())
    bms.add_metric(NDCG())
    bms.add_metric(DCG())
    bms.add_metric(RBP)
    bms.add_metric(RecipRank)

    metrics = bms.measure(recs, split.test)
    scores = metrics.list_metrics()
    stats = metrics.list_summary()
    print(stats)
    for m in bms.metrics:
        assert stats.loc[m.label, "mean"] == approx(scores[m.label].mean())


def test_recs_multi(demo_recs):
    split, recs = demo_recs

    il2 = ItemListCollection(["rep", "user_id"])
    il2.add_from(recs, rep=1)
    il2.add_from(recs, rep=2)

    bms = RunAnalysis()
    bms.add_metric(ListLength())
    bms.add_metric(Precision())
    bms.add_metric(NDCG())
    bms.add_metric(RBP)
    bms.add_metric(RecipRank)

    metrics = bms.measure(il2, split.test)
    scores = metrics.list_metrics()
    stats = metrics.list_summary("rep")
    print(stats)
    for m in bms.metrics:
        assert stats.loc[(1, m.label), "mean"] == approx(scores.loc[1, m.label].mean())
        assert stats.loc[(2, m.label), "mean"] == approx(scores.loc[2, m.label].mean())
