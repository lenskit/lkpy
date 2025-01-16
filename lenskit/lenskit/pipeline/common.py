# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
from typing import Literal

from lenskit.data import ID, ItemList, RecQuery

from ._impl import Pipeline
from .builder import PipelineBuilder
from .components import Component


class RecPipelineBuilder:
    """
    Builder to help assemble common pipeline patterns.

    This is a convenience class; you can always directly assemble a
    :class:`Pipeline` if this class's behavior is inadquate.

    Stability:
        Caller
    """

    _selector: Component
    _scorer: Component
    _ranker: Component
    is_predictor: bool = False
    _predict_transform: Component | None = None
    _fallback: Component | None = None

    def __init__(self):
        from lenskit.basic.candidates import UnratedTrainingItemsCandidateSelector
        from lenskit.basic.topn import TopNRanker

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
        from lenskit.basic.topn import TopNRanker

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
        self, *, transform: Component | None = None, fallback: Component | None = None
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
        from lenskit.basic.composite import FallbackScorer
        from lenskit.basic.history import UserTrainingHistoryLookup

        pipe = PipelineBuilder(name=name)

        query = pipe.create_input("query", RecQuery, ID, ItemList)

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
                pipe.alias("rating-predictor", rater)

        rank = pipe.add_component("ranker", self._ranker, items=n_score, n=n_n)
        pipe.alias("recommender", rank)
        pipe.default_component("recommender")

        return pipe.build()


def topn_pipeline(
    scorer: Component,
    *,
    predicts_ratings: bool | Literal["raw"] = False,
    n: int = -1,
    name: str | None = None,
) -> Pipeline:
    """
    Create a pipeline that produces top-N recommendations using a scoring model.

    Stability:
        Caller

    Args:
        scorer:
            The scorer to use in the pipeline (it will added with the component
            name ``scorer``, see :ref:`pipeline-names`).
        predicts_ratings:
            If ``True``, make set up to predict ratings (``rating-predictor``),
            using ``scorer`` with a fallback of :class:`BiasScorer`; if
            ``"raw"``, use ``scorer`` directly with no fallback.
        n:
            The recommendation list length to configure in the pipeline.
        name:
            The pipeline name.
    """
    from lenskit.basic.bias import BiasScorer

    builder = RecPipelineBuilder()
    builder.scorer(scorer)
    builder.ranker(n=n)
    if predicts_ratings == "raw":
        builder.predicts_ratings()
    elif predicts_ratings:
        builder.predicts_ratings(fallback=BiasScorer())

    return builder.build(name)


def predict_pipeline(
    scorer: Component,
    *,
    fallback: bool | Component = True,
    n: int = -1,
    name: str | None = None,
) -> Pipeline:
    """
    Create a pipeline that predicts ratings, but does **not** include any
    ranking capabilities.  Mostly userful for testing and historical purposes.
    The resulting pipeline **must** be called with an item list.

    Stability:
        Caller

    Args:
        scorer:
            The scorer to use in the pipeline (it will added with the component
            name ``scorer``, see :ref:`pipeline-names`).
        fallback:
            Whether to use a fallback predictor when the scorer cannot score.
            When configured, the `scorer` node is the scorer, and the
            `rating-predictor` node applies the fallback.
        n:
            The recommendation list length to configure in the pipeline.
        name:
            The pipeline name.
    """
    from lenskit.basic.bias import BiasScorer
    from lenskit.basic.composite import FallbackScorer
    from lenskit.basic.history import UserTrainingHistoryLookup

    pipe = PipelineBuilder(name=name)

    query = pipe.create_input("query", RecQuery, ID, ItemList)
    items = pipe.create_input("items", ItemList)

    lookup = pipe.add_component("history-lookup", UserTrainingHistoryLookup(), query=query)

    score = pipe.add_component("scorer", scorer, query=lookup, items=items)

    if fallback is True:
        fallback = BiasScorer()

    if fallback is False:
        pipe.alias("rating-predictor", score)
    else:
        backup = pipe.add_component("fallback-predictor", fallback, query=lookup, items=items)
        pipe.add_component("rating-predictor", FallbackScorer(), primary=score, fallback=backup)

    pipe.default_component("rating-predictor")

    return pipe.build()
