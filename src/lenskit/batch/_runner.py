# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Sized
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, NamedTuple, TypeAlias, overload

import pandas as pd

from lenskit.data import (
    ID,
    GenericKey,
    ItemList,
    ItemListCollection,
    QueryIDKey,
    RecQuery,
    UserIDKey,
)
from lenskit.logging import Stopwatch, get_logger, item_progress
from lenskit.parallel import invoker
from lenskit.pipeline import Pipeline, PipelineProfiler
from lenskit.pipeline._profiling import ProfileSink

from ._results import BatchResults

_log = get_logger(__name__)

ItemSource: TypeAlias = None | Literal["test-items"]
"""
Types of items that can be returned.
"""


class BatchRequest(NamedTuple):
    """
    A single request for the batch inference runner.
    """

    query: RecQuery
    items: ItemList | None = None


@dataclass
class InvocationSpec:
    """
    Specification for a single pipeline invocation, to record one or more
    pipeline component outputs for a test user.
    """

    name: str
    "A name for this invocation."
    components: dict[str, str]
    "The names of pipeline components to measure and return, mapped to their output names."
    items: ItemSource = None
    "The target or candidate items (if any) to provide to the recommender."
    extra_inputs: dict[str, Any] = field(default_factory=dict)
    "Additional inputs to pass to the pipeline."


class BatchPipelineRunner:
    """
    Apply a pipeline to a collection of test users.

    Stability:
        Caller

    Argss:
        pipeline:
            The pipeline to evaluate.
        n_jobs:
            The number of parallel processes to use, or ``None`` for the
            default (defined by :func:`lenskit.parallel.config.initialize`).
    """

    n_jobs: int | Literal["ray"] | None
    profiler: PipelineProfiler | None
    invocations: list[InvocationSpec]

    def __init__(
        self,
        *,
        n_jobs: int | Literal["ray"] | None = None,
        profiler: PipelineProfiler | None = None,
    ):
        self.n_jobs = n_jobs
        self.profiler = profiler
        self.invocations = []

    def add_invocation(self, inv: InvocationSpec):
        self.invocations.append(inv)

    def score(self, component: str = "scorer", *, output: str = "scores"):
        """
        Request the batch run to generate test item scores.

        Args:
            component:
                The name of the rating predictor component to run.
            output:
                The name of the results in the output dictionary.
        """
        self.add_invocation(InvocationSpec("score", {component: output}, "test-items"))

    def predict(self, component: str = "rating-predictor", *, output: str = "predictions"):
        """
        Request the batch run to generate test item rating predictions.  It is identical
        to :meth:`score` but with different defaults.

        Args:
            component:
                The name of the rating predictor component to run.
            output:
                The name of the results in the output dictionary.
        """
        return self.score(component, output=output)

    def recommend(
        self, component: str = "recommender", *, output: str = "recommendations", **extra: Any
    ):
        """
        Request the batch run to generate recomendations.

        Args:
            component:
                The name of the recommender component to run.
            output:
                The name of the results in the output dictionary.
            extra:
                Extra inputs to the recommender. A common option is ``n``, the
                number of recommendations to return (a default may be baked into
                the pipeline).
        """
        self.add_invocation(InvocationSpec("recommend", {component: output}, extra_inputs=extra))

    @overload
    def run(
        self,
        pipeline: Pipeline,
        queries: Iterable[RecQuery]
        | Iterable[tuple[RecQuery, ItemList]]
        | Iterable[ID | GenericKey]
        | ItemListCollection[GenericKey]
        | Mapping[ID, ItemList]
        | pd.DataFrame,
    ) -> BatchResults: ...
    @overload
    def run(
        self,
        pipeline: Pipeline,
        *,
        test_data: Iterable[ID | GenericKey]
        | ItemListCollection[GenericKey]
        | Mapping[ID, ItemList]
        | pd.DataFrame,
    ) -> BatchResults: ...
    def run(
        self,
        pipeline: Pipeline,
        queries: Iterable[RecQuery]
        | Iterable[tuple[RecQuery, ItemList]]
        | Iterable[ID | GenericKey]
        | ItemListCollection[GenericKey]
        | Mapping[ID, ItemList]
        | pd.DataFrame
        | None = None,
        *,
        test_data: Iterable[ID | GenericKey]
        | ItemListCollection[GenericKey]
        | Mapping[ID, ItemList]
        | pd.DataFrame
        | None = None,
    ) -> BatchResults:
        """
        Run the pipeline and return its results.

        Args:
            pipeline:
                The pipeline to run.
            queries:
                The collection of test queries use.  See :ref:`batch-queries`
                for details on the various input formats.

        Returns:
            The results, as a nested dictionary.  The outer dictionary maps
            component output names to inner dictionaries of result data.
        """
        if test_data is not None:  # pragma: nocover
            if queries is not None:
                raise RuntimeError("cannot specify both queries and test_data=")
            queries = test_data
            warnings.warn(
                "the test_data parameter is renamed to queries", DeprecationWarning, stacklevel=2
            )

        if queries is None:  # pragma: nocover
            raise RuntimeError("no queries specified")

        prof = self.profiler
        if prof is not None:
            prof = prof.multiprocess()

        key_type, q_iter, nq = _normalize_query_input(queries)

        log = _log.bind(name=pipeline.name, n_queries=nq, n_jobs=self.n_jobs)
        log.info("beginning batch run")

        with (
            invoker(
                (pipeline, self.invocations, prof), _run_pipeline, n_jobs=self.n_jobs
            ) as worker,
            item_progress("Inference", nq) as progress,
        ):
            # release our reference, will sometimes free the pipeline memory in this process
            del pipeline
            results = BatchResults(key_type)
            timer = Stopwatch()
            n = 0
            for key, outs in worker.map(q_iter):
                n += 1
                for cn, cr in outs.items():
                    results.add_result(cn, key, cr)
                progress.update()
            timer.stop()

            rate_ms = timer.elapsed() / n * 1000
            log.info("finished running in %s", timer, time_per_query="{:.1f}ms".format(rate_ms))

        return results


def _normalize_query_input(
    queries: Iterable[RecQuery]
    | Iterable[tuple[RecQuery, ItemList]]
    | Iterable[ID | GenericKey]
    | ItemListCollection[GenericKey]
    | Mapping[ID, ItemList]
    | pd.DataFrame,
) -> tuple[type[Any], Iterable[tuple[RecQuery, ItemList | None]], int | None]:
    if isinstance(queries, pd.DataFrame):
        warnings.warn(
            "use an item list collection instead of a DataFrame (LKW-BATCHIN)",
            DeprecationWarning,
            stacklevel=2,
        )
        queries = ItemListCollection.from_df(queries)

    elif isinstance(queries, Mapping):
        warnings.warn(
            "query mappings are ambiguous and deprecated, use query lists (LKW-BATCHIN)",
            DeprecationWarning,
            stacklevel=2,
        )
        queries = ItemListCollection.from_dict(queries, "user_id")  # type: ignore

    if isinstance(queries, ItemListCollection):
        return queries.key_type, _ilc_queries(queries), len(queries)

    n = None
    if isinstance(queries, Sized):
        n = len(queries)

    q_iter = iter(queries)
    try:
        q_first = next(q_iter)
    except StopIteration:
        return tuple, [], 0

    fbr = _make_br(q_first)
    if fbr.query.query_id is not None:
        kt = QueryIDKey
    elif fbr.query.user_id is not None:
        kt = UserIDKey
    else:
        raise ValueError("query must have one of query_id, user_id")

    return kt, _iter_queries(q_first, q_iter), n


def _ilc_queries(queries: ItemListCollection):
    for q, items in queries.items():
        query = RecQuery(
            user_id=getattr(q, "user_id", None),
            query_id=getattr(q, "query_id", None),
        )
        yield BatchRequest(query, items)


def _iter_queries(
    first: RecQuery | tuple[RecQuery, ItemList] | ID | GenericKey,
    rest: Iterator[RecQuery | tuple[RecQuery, ItemList] | ID | GenericKey],
) -> Iterable[BatchRequest]:
    yield _make_br(first)
    for item in rest:
        yield _make_br(item)


def _make_br(q: RecQuery | tuple[RecQuery, ItemList] | ID | GenericKey) -> BatchRequest:
    if isinstance(q, RecQuery):
        return BatchRequest(q)
    elif isinstance(q, tuple):
        if isinstance(q[0], RecQuery):
            q, items = q
            return BatchRequest(q, items)  # type: ignore
        elif hasattr(q, "user_id"):
            # we have a named tuple with user IDs
            q = RecQuery(user_id=getattr(q, "user_id"))
            return BatchRequest(q)
        else:
            warnings.warn(
                "bare tuples are ambiguous and will be unsupported in 2026 (LKW-BATCHIN)",
                DeprecationWarning,
                stacklevel=3,
            )
            q = RecQuery(user_id=q)  # type: ignore
            return BatchRequest(q)
    else:
        q = RecQuery(user_id=q)
        return BatchRequest(q)


def _ensure_key(key: ID | tuple[ID, ...]):
    if isinstance(key, tuple):
        return key
    else:
        return UserIDKey(key)


def _run_pipeline(
    ctx: tuple[Pipeline, list[InvocationSpec], ProfileSink | None],
    req: BatchRequest,
) -> tuple[GenericKey, dict[str, object]]:
    pipeline, invocations, profiler = ctx
    query, test_items = req
    if query.query_id is not None:
        key = QueryIDKey(query.query_id)
    elif query.user_id is not None:
        key = UserIDKey(query.user_id)
    else:
        raise RuntimeError("query must have one of query_id, user_id")

    result = {}

    _log.debug("running pipeline", name=pipeline.name, key=key)
    for inv in invocations:
        inputs: dict[str, Any] = {"query": query}
        match inv.items:
            case "test-items" if test_items is not None:
                inputs["items"] = test_items

        inputs.update(inv.extra_inputs)

        nodes = inv.components.keys()
        outs = pipeline.run_all(*nodes, _profile=profiler, **inputs)
        for cname, oname in inv.components.items():
            result[oname] = outs[cname]

    return key, result  # type: ignore
