# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.basic import (
    FallbackScorer,
    UnratedTrainingItemsCandidateSelector,
    UserTrainingHistoryLookup,
)
from lenskit.basic.topn import TopNRanker
from lenskit.data import EntityId, ItemList, RecQuery

from . import Pipeline
from .components import Component


class RecPipelineBuilder:
    """
    Builder to help assemble common pipeline patterns.

    This is a convenience class; you can always directly assemble a
    :class:`Pipeline` if this class's behavior is inadquate.
    """

    _selector: Component
    _scorer: Component
    _ranker: Component
    is_predictor: bool = False
    _predict_transform: Component | None = None
    _fallback: Component | None = None

    def __init__(self):
        self._selector = UnratedTrainingItemsCandidateSelector()
        self._ranker = TopNRanker()

    def scorer(self, score: Component):
        """
        Specify the scoring model.
        """
        self._scorer = score

    def ranker(self, rank: Component | None = None, *, n: int = -1):
        """
        Specify the ranker to use.  If ``None``, sets up a :class:`TopNRanker`
        with ``n=n``.
        """
        if rank is None:
            self._ranker = TopNRanker(n=n)
        else:
            self._ranker = rank

    def candidate_selector(self, sel: Component):
        """
        Specify the candidate selector component.  The component should accept
        a query as its input and return an item list.
        """
        self._selector = sel

    def predicts_ratings(
        self, transform: Component | None = None, *, fallback: Component | None = None
    ):
        """
        Specify that this pipeline will predict ratings, optionally providing a
        rating transformer and fallback scorer for the rating predictions.

        Args:
            transform:
                A component to transform scores prior to returning them.  If
                supplied, it will be applied to both primary scores and fallback
                scores.
            fallback:
                A fallback scorer to use when the primary scorer cannot score an
                item. The fallback should accept ``query`` and ``items`` inputs,
                and return an item list.
        """
        self.is_predictor = True
        self._predict_transform = transform
        self._fallback = fallback

    def build(self, name: str | None = None) -> Pipeline:
        """
        Build the specified pipeline.
        """
        pipe = Pipeline(name=name)

        query = pipe.create_input("query", RecQuery, EntityId, ItemList)

        items = pipe.create_input("items", ItemList)
        n_n = pipe.create_input("n", int, None)

        lookup = pipe.add_component("history-lookup", UserTrainingHistoryLookup(), query=query)
        cand_sel = pipe.add_component("candidate-selector", self._selector, query=lookup)
        candidates = pipe.use_first_of("candidates", items, cand_sel)

        n_score = pipe.add_component("scorer", self._scorer, query=lookup, items=candidates)
        if self.is_predictor:
            if self._fallback is not None:
                fb = pipe.add_component(
                    "fallback-predictor", self._fallback, query=lookup, items=candidates
                )
                rater = pipe.add_component(
                    "rating-merger", FallbackScorer(), scores=n_score, backup=fb
                )
            else:
                rater = n_score

            if self._predict_transform:
                pipe.add_component(
                    "rating-predictor", self._predict_transform, query=query, items=rater
                )
            else:
                pipe.alias("rating-predictor", n_score)

        rank = pipe.add_component("ranker", self._ranker, items=n_score, n=n_n)
        pipe.alias("recommender", rank)

        return pipe


def topn_pipeline(scorer: Component, *, predicts_ratings: bool = False, n: int = -1) -> Pipeline:
    """
    Create a pipeline that produces top-N recommendations using the specified
    scorer.  The scorer should have the following call signature::

        def scorer(user: UserHistory, items: ItemList) -> pd.Series: ...

    Args:
        scorer:
            The scorer to use in the pipeline (it will added with the component
            name ``scorer``, see :ref:`pipeline-names`).
        predicts_ratings:
            If ``True``, make ``rating-predictor`` an alias for ``scorer`` so that
            evaluation components know this pipeline can predict ratings.
        n:
            The recommendation list length to configure in the pipeline.
    """
    builder = RecPipelineBuilder()
    builder.scorer(scorer)
    builder.ranker(n=n)
    if predicts_ratings:
        builder.predicts_ratings()

    return builder.build()
