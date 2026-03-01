# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, Mapping, TypeAlias, overload

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
from lenskit.parallel import get_parallel_config, invoker, is_free_threaded
from lenskit.pipeline import Pipeline, PipelineProfiler
from lenskit.pipeline._profiling import ProfileSink

from ._queries import BatchRequest, normalize_query_input
from ._results import BatchResultRow, BatchResults

_log = get_logger(__name__)

ItemSource: TypeAlias = None | Literal["test-items"]
"""
Types of items that can be returned.
"""


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
            The number of parallel threads to use, or ``None`` for default
            defined by LensKit configuration and environment variables (see
            :ref:`parallel-config`).
        batch_size:
            The batch size for multiprocess execution.
    """

    n_jobs: int | Literal["ray"]
    batch_size: int
    profiler: PipelineProfiler | None
    invocations: list[InvocationSpec]

    def __init__(
        self,
        *,
        n_jobs: int | Literal["ray"] | None = None,
        profiler: PipelineProfiler | None = None,
        batch_size: int = 200,
    ):
        if n_jobs is None:
            n_jobs = get_parallel_config().batch_jobs
        self.n_jobs = n_jobs
        self.batch_size = batch_size

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

        .. note::

            The runner does **not** guarantee that results are in the same order
            as the original inputs.

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

        key_type, q_iter, nq = normalize_query_input(queries)

        log = _log.bind(name=pipeline.name, n_queries=nq, n_jobs=self.n_jobs)

        log.info("beginning batch run")

        with closing(self._run_results(pipeline, prof, q_iter)) as tasks:
            with item_progress("Inference", nq) as progress:
                # release our reference, will sometimes free the pipeline memory in this process
                del pipeline
                results = BatchResults(key_type)
                timer = Stopwatch()
                n = 0
                for key, outs in tasks:
                    n += 1
                    for cn, cr in outs.items():
                        results.add_result(cn, key, cr)
                    progress.update()
                timer.stop()

                rate_ms = timer.elapsed() / n * 1000
                log.info("finished running in %s", timer, time_per_query="{:.1f}ms".format(rate_ms))

        return results

    def _run_results(
        self, pipeline: Pipeline, profiler: ProfileSink | None, queries: Iterable[BatchRequest]
    ) -> Generator[BatchResultRow]:
        if is_free_threaded() and isinstance(self.n_jobs, int) and self.n_jobs > 1:
            return self._threaded_results(pipeline, profiler, queries)
        else:
            return self._invoker_results(pipeline, profiler, queries)

    def _invoker_results(
        self,
        pipeline: Pipeline,
        profiler: ProfileSink | None,
        queries: Iterable[BatchRequest],
    ) -> Generator[BatchResultRow]:
        with invoker(
            (pipeline, self.invocations, profiler), run_pipeline, n_jobs=self.n_jobs
        ) as worker:
            yield from worker.map(queries)

    def _threaded_results(
        self,
        pipeline: Pipeline,
        profiler: ProfileSink | None,
        queries: Iterable[BatchRequest],
    ) -> Generator[BatchResultRow]:
        assert isinstance(self.n_jobs, int)
        func = partial(run_pipeline, (pipeline, self.invocations, profiler))
        _log.info("using thread pool with %d threads", self.n_jobs)
        if not is_free_threaded(require_active=True):
            _log.warn("using thread pool but Python GIL is enabled, throughput will suffer")
        with ThreadPoolExecutor(self.n_jobs, "lk-batch") as pool:
            yield from pool.map(
                func,
                queries,
                chunksize=self.batch_size,
                buffersize=self.n_jobs * self.batch_size * 1,
            )


def run_pipeline(
    ctx: tuple[Pipeline, list[InvocationSpec], ProfileSink | None],
    req: BatchRequest,
) -> BatchResultRow:
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
