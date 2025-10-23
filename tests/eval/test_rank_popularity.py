# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from pytest import approx

from lenskit.data import Dataset, ItemList
from lenskit.metrics.ranking import MeanPopRank
from lenskit.testing import DemoRecs, demo_recs  # noqa: F401


def _test_mpr(ml_ds, items, rel, **k):
    recs = ItemList(items, ordered=True)
    truth = ItemList(rel)
    return MeanPopRank(ml_ds, **k).measure_list(recs, truth)


def test_mpr_empty_none(ml_ds: Dataset):
    mpr = _test_mpr(ml_ds, [], [1, 3])
    assert np.isnan(mpr)


def test_mpr_label(ml_ds: Dataset):
    mpr = MeanPopRank(ml_ds)
    assert mpr.label == "MeanPopRank"

    mpr = MeanPopRank(ml_ds, n=10)
    assert mpr.label == "MeanPopRank@10"


def test_mpr_single_max(ml_ds: Dataset):
    items = ml_ds.item_stats().sort_values("count", ascending=False)
    mpr = _test_mpr(ml_ds, [items.index[0]], [1, 3])
    assert mpr == approx(1.0)


def test_mpr_single_min(ml_ds: Dataset):
    items = ml_ds.item_stats().sort_values("count", ascending=False)
    mpr = _test_mpr(ml_ds, [items.index[-1]], [1, 3])
    assert mpr == approx(0.0)


def test_mpr_min_max(ml_ds: Dataset):
    items = ml_ds.item_stats().sort_values("count", ascending=False)
    mpr = _test_mpr(ml_ds, [items.index[-1], items.index[0]], [1, 3])
    assert mpr == approx(0.5)


def test_mpr_random(ml_ds: Dataset, rng: np.random.Generator):
    items = rng.choice(ml_ds.items.ids(), 100, replace=False)
    mpr = _test_mpr(ml_ds, items, [])

    stats = ml_ds.item_stats()
    counts = stats["count"]
    counts = counts[counts > 0]
    ranks = counts.rank(method="average", ascending=True) / len(counts)
    assert mpr == approx(ranks.reindex(items, fill_value=0).mean(), rel=1.0e-3)
