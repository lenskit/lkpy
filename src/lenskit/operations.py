# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Top-level LensKit functions for recommender operations.
"""

from __future__ import annotations

from lenskit.data import ItemList, QueryInput
from lenskit.data.types import IDSequence
from lenskit.pipeline import Pipeline, PipelineProfiler


def recommend(
    pipeline: Pipeline,
    query: QueryInput,
    n: int | None = None,
    items: ItemList | IDSequence | None = None,
    *,
    component: str = "recommender",
    profiler: PipelineProfiler | None = None,
) -> ItemList:
    """
    Generate recommendations for a user or query.  This calls the specified
    pipeline component (the ``'recommender'`` by default) and returns the
    resulting item list.

    Stability:
        full
    Args:
        pipeline:
            The pipeline to run.
        query:
            The user ID or other query data for the recommendation.
        n:
            The number of items to recommend.
        items:
            The candidate items, or ``None`` to use the pipeline's default
            candidate selector.
        component:
            The name of the component implementing the recommender.
        profiler:
            A profiler for profiling this pipeline run.

    Returns:
        The recommended items as an ordered item list.
    """

    node = pipeline.node(component)
    if items is not None and not isinstance(items, ItemList):
        items = ItemList(items)
    res = pipeline.run(node, query=query, n=n, items=items, _profile=profiler)
    if not isinstance(res, ItemList):
        raise TypeError("recommender pipeline did not return an item list")

    return res


def score(
    pipeline: Pipeline,
    query: QueryInput,
    items: ItemList | IDSequence,
    *,
    component: str = "scorer",
    profiler: PipelineProfiler | None = None,
) -> ItemList:
    """
    Score items with respect to a user or query.  This calls the specified
    pipeline component (the ``'scorer'`` by default) and returns the resulting
    item list.

    Stability:
        Full
    Args:
        pipeline:
            The pipeline to run.
        query:
            The user ID or other query data for the recommendation.
        items:
            The candidate items, or ``None`` to use the pipeline's default
            candidate selector.
        component:
            The name of the component implementing the scorer.
    Returns:
        The items with scores in the ``score`` field.
    """

    node = pipeline.node(component)
    if items is not None and not isinstance(items, ItemList):
        items = ItemList(items)
    res = pipeline.run(node, query=query, items=items, _profile=profiler)
    if not isinstance(res, ItemList):
        raise TypeError("scorer pipeline did not return an item list")

    return res


def predict(
    pipeline: Pipeline,
    query: QueryInput,
    items: ItemList | IDSequence,
    *,
    component: str = "rating-predictor",
    profiler: PipelineProfiler | None = None,
) -> ItemList:
    """
    Predict ratings for items.  This is exactly like :func:`score`, except it
    defaults to the ``'rating-predictor'`` component.  In a standard pipeline,
    the rating predictor may have additional configuration such as fallbacks or
    transformations to ensure every item is scored and the scores are valid
    rating predictions; the scorer typically returns raw scores.

    Stability:
        Full
    """

    return score(pipeline, query, items, component=component, profiler=profiler)
