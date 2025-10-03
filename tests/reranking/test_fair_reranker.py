# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

import pytest

from lenskit.basic import PopScorer
from lenskit.data import DatasetBuilder, ItemList, from_interactions_df
from lenskit.logging import get_logger
from lenskit.pipeline import RecPipelineBuilder
from lenskit.reranking.fair import FairReranker
from lenskit.training import TrainingOptions

_log = get_logger(__name__)


def make_dataset(item_ids, protected_flags):
    """Helper to build a minimal Dataset with protected attributes."""

    # dummy interactions dataset
    interactions = pd.DataFrame(
        {
            "user_id": range(len(item_ids)),
            "item_id": item_ids,
        }
    )

    protected_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "protected": protected_flags,
        }
    )

    ds = from_interactions_df(interactions)
    builder = DatasetBuilder(ds)

    builder.add_scalar_attribute("item", "protected", protected_df)
    ds = builder.build()
    return ds


def test_compute_m_list():
    k, p, alpha = 10, 0.5, 0.1
    reranker = FairReranker(k=k, p=p, alpha=alpha)
    m_list = reranker._compute_m_list(k, p, alpha)
    # check Table 2 of Zehlike et al. (2017)
    assert m_list[3] == 1
    assert m_list[6] == 2
    assert m_list[8] == 3
    assert len(m_list) == k


def test_compute_blocks():
    reranker = FairReranker(k=9, p=0.5, alpha=0.1)
    m_list = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3])
    blocks = reranker._compute_blocks(m_list)

    assert list(blocks) == [4, 3, 2]


def test_rerank_small_list():
    items = np.arange(6, 15)
    protected = [False, False, False, True, False, True, True, False, True]
    data = make_dataset(items, protected)

    reranker = FairReranker(k=9, p=0.5, alpha=0.1)
    reranker.train(data)

    sample_vanilla_list = ItemList(item_ids=items, ordered=True)
    fair_reranked = reranker(sample_vanilla_list)

    # Check that protected counts meet m_list thresholds
    ids = fair_reranked.ids()
    prot_flags = data.entities("item").attribute("protected").pandas().reindex(ids)

    counts = prot_flags.cumsum()
    assert np.all(counts >= reranker.m_list_[: len(ids)])


@pytest.mark.parametrize(
    "protected",
    [
        [False] * 6,  # all unprotected
        [True] * 6,  # all protected
    ],
)
def test_all_unprotected_items(protected):
    items = np.arange(6, 12)
    data = make_dataset(items, protected)

    reranker = FairReranker(k=6, p=0.5, alpha=0.1)
    reranker.train(data)

    sample_vanilla_list = ItemList(item_ids=items, ordered=True)
    fair_reranked = reranker(sample_vanilla_list)

    # Nothing to rerank, should return same order
    assert np.array_equal(fair_reranked.ids(), items)


@pytest.mark.parametrize(
    "n,p,alpha", [(20, 0.5, 0.2), (50, 0.5, 0.1), (100, 0.1, 0.3), (150, 0.9, 0.1)]
)
def test_randomized_reranking(n, p, alpha):
    rng = np.random.default_rng(42)
    items = rng.permutation(n)
    protected = rng.random(n) < p
    data = make_dataset(items, protected)

    reranker = FairReranker(k=n, p=p, alpha=alpha)
    reranker.train(data)

    sample_vanilla_list = ItemList(item_ids=items, ordered=True)
    fair_reranked = reranker(sample_vanilla_list)

    # verify that fairness constraints are enforced
    counts = np.cumsum([iid in items[protected] for iid in fair_reranked.ids()])
    assert np.all(counts >= reranker.m_list_[: len(counts)])


def test_pipeline_with_fair_reranker():
    """
    End-to-end test: PopScorer -> FairReranker in a RecPipelineBuilder.
    Ensures reranker can work as part of the pipeline.
    """
    items = np.arange(5, 15)
    protected = [False, False, False, False, False, True, False, False, True, True]
    ds = make_dataset(items, protected)

    builder = RecPipelineBuilder()
    builder.scorer(PopScorer)
    builder.reranker("fair", k=9, p=0.5, alpha=0.1)  # attach reranker
    pipe = builder.build()
    pipe.train(ds, TrainingOptions())

    # get recommendations for a user
    recs = pipe.run("recommender", query=0)
    assert isinstance(recs, ItemList)
    ids = recs.ids()

    # verify that fairness constraints are enforced
    prot_flags = ds.entities("item").attribute("protected").pandas().reindex(ids)
    counts = prot_flags.cumsum()
    reranker = pipe.node("reranker").component
    assert np.all(counts >= reranker.m_list_[: len(ids)])
