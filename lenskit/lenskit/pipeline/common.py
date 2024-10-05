# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.data import EntityId, ItemList, RecQuery

from . import Pipeline
from .components import Component


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
    from lenskit.basic import UnratedTrainingItemsCandidateSelector, UserTrainingHistoryLookup
    from lenskit.basic.topn import TopNRanker

    pipe = Pipeline()

    query = pipe.create_input("query", RecQuery, EntityId, ItemList)
    items = pipe.create_input("items", ItemList)
    n_n = pipe.create_input("n", int, None)

    lookup = pipe.add_component("history-lookup", UserTrainingHistoryLookup(), query=query)
    cand_sel = pipe.add_component(
        "candidate-selector", UnratedTrainingItemsCandidateSelector(), query=lookup
    )
    candidates = pipe.use_first_of("candidates", items, cand_sel)

    n_score = pipe.add_component("scorer", scorer, query=lookup, items=candidates)
    if predicts_ratings:
        pipe.alias("rating-predictor", n_score)

    _rank = pipe.add_component("ranker", TopNRanker(n=n), items=n_score, n=n_n)

    return pipe
