# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Literal, NamedTuple

from pydantic import JsonValue

from lenskit.data import ID, ItemList, RecQuery
from lenskit.pipeline.config import PipelineOptions

from ._impl import Pipeline
from .builder import PipelineBuilder
from .components import Component, ComponentConstructor, Placeholder


class CompRec(NamedTuple):
    component: Component | ComponentConstructor
    config: object | None = None


class RecPipelineBuilder:
    """
    Builder to help assemble common pipeline patterns.

    This is a convenience class; you can always directly assemble a
    :class:`Pipeline` if this class's behavior is inadquate.

    Stability:
        Caller
    """

    _selector: CompRec
    _scorer: CompRec
    _ranker: CompRec
    is_predictor: bool = False
    _predict_transform: Component | None = None
    _fallback: Component | None = None
    _reranker: CompRec | None = None

    def __init__(self):
        from lenskit.basic.candidates import TrainingItemsCandidateSelector
        from lenskit.basic.topn import TopNRanker

        self._selector = CompRec(TrainingItemsCandidateSelector)
        self._ranker = CompRec(TopNRanker)

    def scorer(self, score: Component | ComponentConstructor, config: object | None = None):
        """
        Specify the scoring model.
        """
        self._scorer = CompRec(score, config)

    def ranker(
        self,
        rank: Component | ComponentConstructor | None = None,
        config: object | None = None,
        *,
        n: int | None = None,
    ):
        """
        Specify the ranker to use.  If ``None``, sets up a :class:`TopNRanker`
        with ``n=n``.
        """
        from lenskit.basic.topn import TopNConfig, TopNRanker

        if rank is None:
            self._ranker = CompRec(TopNRanker, TopNConfig(n=n))
        else:
            self._ranker = CompRec(rank, config)

    def candidate_selector(
        self, sel: Component | ComponentConstructor, config: object | None = None
    ):
        """
        Specify the candidate selector component.  The component should accept
        a query as its input and return an item list.
        """
        self._selector = CompRec(sel, config)

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
        cand_sel = pipe.add_component(
            "candidate-selector", self._selector.component, self._selector.config, query=lookup
        )
        candidates = pipe.use_first_of("candidates", items, cand_sel)

        n_score = pipe.add_component(
            "scorer", self._scorer.component, self._scorer.config, query=lookup, items=candidates
        )
        if self.is_predictor:
            if self._fallback is not None:
                fb = pipe.add_component(
                    "fallback-predictor", self._fallback, query=lookup, items=candidates
                )
                rater = pipe.add_component(
                    "rating-merger", FallbackScorer(), primary=n_score, backup=fb
                )
            else:
                rater = n_score

            if self._predict_transform:
                pipe.add_component(
                    "rating-predictor", self._predict_transform, query=query, items=rater
                )
            else:
                pipe.alias("rating-predictor", rater)

        rank = pipe.add_component(
            "ranker", self._ranker.component, self._ranker.config, items=n_score, n=n_n
        )

        # If a reranker is configured, attach it
        if self._reranker is not None:
            rerank = pipe.add_component(
                "reranker", self._reranker.component, self._reranker.config, items=rank, n=n_n
            )
            pipe.alias("recommender", rerank)
            pipe.default_component("recommender")
        else:
            pipe.alias("recommender", rank)
            pipe.default_component("recommender")

        return pipe.build()

    def reranker(self, reranker: Component | ComponentConstructor, config: object | None = None):
        """
        Specify a reranker to add to the pipeline.

        Args:
            reranker:
                The reranker to use in the pipeline.
            config:
                Configuration parameters to initialize the reranker if a reranker is specified.
        """
        self._reranker = CompRec(reranker, config)
        return self


def topn_builder(
    name: str | None = None,
    options: PipelineOptions | dict[str, JsonValue] | None = None,
) -> PipelineBuilder:
    """
    Construct a new pipeline builder set up for top-*N*.

    This is used as the "std:topn" base.

    Args:
        name:
            The pipeline name.
        options:
            The pipeline options to configure the base pipeline.
    """

    from lenskit.basic.candidates import TrainingItemsCandidateSelector
    from lenskit.basic.history import UserTrainingHistoryLookup
    from lenskit.basic.topn import TopNRanker

    options = PipelineOptions.model_validate(options or {})

    pipe = PipelineBuilder(name=name)
    query = pipe.create_input("query", RecQuery, ID, ItemList)
    items = pipe.create_input("items", ItemList)
    n_n = pipe.create_input("n", int, None)

    lookup = pipe.add_component("history-lookup", UserTrainingHistoryLookup, query=query)
    cand_sel = pipe.add_component(
        "candidate-selector", TrainingItemsCandidateSelector, query=lookup
    )
    candidates = pipe.use_first_of("candidates", items, cand_sel)

    n_score = pipe.add_component("scorer", Placeholder, query=lookup, items=candidates)

    rank = pipe.add_component(
        "ranker", TopNRanker, {"n": options.default_length}, items=n_score, n=n_n
    )
    pipe.alias("recommender", rank)
    pipe.default_component("recommender")
    return pipe


def topn_predict_builder(
    name: str | None = None, options: PipelineOptions | dict[str, JsonValue] | None = None
):
    """
    Construct a new pipeline builder set up for top-*N* with rating predictions.

    This is used as the "std:topn-predict" base.  It respects the
    :attr:`~PipelineOptions.fallback_predictor` option, which defaults to
    ``True``.

    Args:
        name:
            The pipeline name.
        options:
            The pipeline options to configure the base pipeline.
    """
    from lenskit.basic import BiasScorer, FallbackScorer

    options = PipelineOptions.model_validate(options or {})

    pipe = topn_builder(name, options)
    lookup = pipe.node("history-lookup")
    candidates = pipe.node("candidates")
    scorer = pipe.node("scorer")

    if options.fallback_predictor is False:
        pipe.alias("rating-predictor", scorer)
    else:
        fb = pipe.add_component("fallback-predictor", BiasScorer, query=lookup, items=candidates)
        rater = pipe.add_component("rating-merger", FallbackScorer, primary=scorer, backup=fb)
        pipe.alias("rating-predictor", rater)

    return pipe


def topn_pipeline(
    scorer: Component | ComponentConstructor,
    config: object | None = None,
    *,
    predicts_ratings: bool | Literal["raw"] = False,
    n: int | None = None,
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
    builder.scorer(scorer, config)
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
            The recommendation list length to configure in the pipeline.  This
            parameter is ignored, and will be removed in 2026.
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
        pipe.add_component("rating-predictor", FallbackScorer(), primary=score, backup=backup)

    pipe.default_component("rating-predictor")

    return pipe.build()
