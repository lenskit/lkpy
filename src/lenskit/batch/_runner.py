# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypeAlias

import pandas as pd

from lenskit.data import ID, GenericKey, ItemList, ItemListCollection, UserIDKey
from lenskit.logging import Stopwatch, get_logger, item_progress
from lenskit.parallel import invoker
from lenskit.pipeline import Pipeline

from ._results import BatchResults

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
            The number of parallel processes to use, or ``None`` for the
            default (defined by :func:`lenskit.parallel.config.initialize`).
    """

    n_jobs: int | None
    invocations: list[InvocationSpec]

    def __init__(self, *, n_jobs: int | None = None):
        self.n_jobs = n_jobs
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

    def run(
        self,
        pipeline: Pipeline,
        test_data: ItemListCollection[GenericKey]
        | Mapping[ID, ItemList]
        | Iterable[ID | GenericKey]
        | pd.DataFrame,
    ) -> BatchResults:
        """
        Run the pipeline and return its results.

        Args:
            test_data:
                The collection of test data, as an ItemListCollection, a mapping
                of user IDs to test data, or as a sequence of item IDs for
                recommendation.

        Returns:
            The results, as a nested dictionary.  The outer dictionary maps
            component output names to inner dictionaries of result data.  These
            inner dictionaries map user IDs to
        """
        if isinstance(test_data, pd.DataFrame):
            test_data = ItemListCollection.from_df(test_data)

        if isinstance(test_data, ItemListCollection):
            test_iter = test_data.items()
            key_type = test_data.key_type
            n_users = len(test_data)
        elif isinstance(test_data, Mapping):
            key_type = UserIDKey
            test_iter = ((UserIDKey(k), v) for (k, v) in test_data.items())  # type: ignore
            n_users = len(test_data)
        else:
            key_type = UserIDKey
            test_data = list(test_data)
            test_iter = ((_ensure_key(k), None) for k in test_data)
            n_users = len(test_data)

        log = _log.bind(
            name=pipeline.name, hash=pipeline.config_hash, n_queries=n_users, n_jobs=self.n_jobs
        )
        log.info("beginning batch run")

        with (
            invoker((pipeline, self.invocations), _run_pipeline, n_jobs=self.n_jobs) as worker,
            item_progress("Recommending", n_users) as progress,
        ):
            # release our reference, will sometimes free the pipeline memory in this process
            del pipeline
            results = BatchResults(key_type)
            timer = Stopwatch()
            for key, outs in worker.map(test_iter):
                for cn, cr in outs.items():
                    results.add_result(cn, key, cr)
                progress.update()
            timer.stop()

            rate = timer.elapsed() / n_users
            log.info("finished running in %s (%.1fms/user)", timer, rate)

        return results


def _ensure_key(key: ID | tuple[ID, ...]):
    if isinstance(key, tuple):
        return key
    else:
        return UserIDKey(key)


def _run_pipeline(
    ctx: tuple[Pipeline, list[InvocationSpec]],
    req: tuple[GenericKey, ItemList],
) -> tuple[GenericKey, dict[str, object]]:
    pipeline, invocations = ctx
    key, test_items = req

    result = {}

    _log.debug("running pipeline", name=pipeline.name, key=key)
    for inv in invocations:
        inputs: dict[str, Any] = {}
        if hasattr(key, "user_id"):
            inputs["query"] = key.user_id  # type: ignore
        match inv.items:
            case "test-items":
                inputs["items"] = test_items

        inputs.update(inv.extra_inputs)

        nodes = inv.components.keys()
        outs = pipeline.run_all(*nodes, **inputs)
        for cname, oname in inv.components.items():
            result[oname] = outs[cname]

    return key, result  # type: ignore
