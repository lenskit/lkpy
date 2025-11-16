# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Test co-occurrance counting.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_array

from lenskit.data import Dataset


def test_basic_cooc(ml_ds: Dataset, rng: np.random.Generator):
    item_cooc = ml_ds.interactions().co_occurrences("item")
    assert isinstance(item_cooc, coo_array)
    assert item_cooc.shape == (ml_ds.item_count, ml_ds.item_count)
    print("total entries:", item_cooc.nnz)

    item_cooc = item_cooc.tocsr()

    item_tbl = ml_ds.interactions().pandas()
    for i in rng.choice(ml_ds.item_count, 50):
        l1 = item_tbl[item_tbl["item_num"] == i]
        lim = item_tbl[item_tbl["user_num"].isin(l1["user_num"])]
        counts = lim.groupby("item_num")["user_num"].count()
        sp, ep = item_cooc.indptr[i : i + 2]
        assert np.all(item_cooc.data[sp:ep] == counts.loc[item_cooc.indices[sp:ep]].values)


def test_ordered_cooc(ml_ds: Dataset, rng: np.random.Generator):
    item_cooc = ml_ds.interactions().co_occurrences("item", order="timestamp")
    assert isinstance(item_cooc, coo_array)
    assert item_cooc.shape == (ml_ds.item_count, ml_ds.item_count)
    print("total entries:", item_cooc.nnz)

    item_cooc = item_cooc.tocsr()

    item_tbl = ml_ds.interactions().pandas()
    for i in rng.choice(ml_ds.item_count, 50):
        l1 = item_tbl[item_tbl["item_num"] == i]
        lim = pd.merge(item_tbl, l1, on="user_num", suffixes=[".t", ".u"])
        lim = lim[lim["timestamp.t"] > lim["timestamp.u"]]
        counts = lim.groupby("item_num.t")["user_num"].count()
        sp, ep = item_cooc.indptr[i : i + 2]
        assert np.all(item_cooc.data[sp:ep] == counts.loc[item_cooc.indices[sp:ep]].values)
