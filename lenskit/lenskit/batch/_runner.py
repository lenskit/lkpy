# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypeAlias

from lenskit.data import EntityId, ItemList
from lenskit.parallel import invoke_progress, invoker
from lenskit.pipeline import Pipeline
from lenskit.util import Stopwatch

from ._results import BatchResults

_log = logging.getLogger(__name__)

ItemSource: TypeAlias = None | Literal["test-items"]
"""
Types of items that can be returned.
"""
TestData: TypeAlias = Mapping[EntityId, ItemList]
"""
Test data format.
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

    def predict(self, component: str = "rating-predictor", *, output: str = "predictions"):
        """
        Request the batch run to generate test item scores or rating predictins.

        Args:
            component:
                The name of the rating predictor component to run.
            output:
                The name of the results in the output dictionary.
        """
        self.add_invocation(InvocationSpec("predict-ratings", {component: output}, "test-items"))

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
        test_data: TestData,
    ) -> BatchResults:
        """
        Run the pipeline and return its results.

        Args:
            test_data:
                A mapping of user IDs to the test data to run against.

        Returns:
            The results, as a nested dictionary.  The outer dictionary maps
            component output names to inner dictionaries of result data.  These
            inner dictionaries map user IDs to
        """
        n_users = len(test_data)
        _log.info("running pipeline %s for %d queries", pipeline.name, n_users)
        _log.info("pipeline configuration hash: %s", pipeline.config_hash())

        with (
            invoke_progress(_log, "querying", n_users, unit="query") as progress,
            invoker(
                (pipeline, self.invocations), _run_pipeline, n_jobs=self.n_jobs, progress=progress
            ) as worker,
        ):
            # release our reference, will sometimes free the pipeline memory in this process
            del pipeline
            results = BatchResults()
            timer = Stopwatch()
            for user, outs in worker.map(test_data.items()):
                for cn, cr in outs.items():
                    results.add_result(cn, user, cr)
            timer.stop()

            rate = timer.elapsed() / n_users
            _log.info("finished running for %d users in %s (%.1fms/user)", n_users, timer, rate)

        return results


def _run_pipeline(
    ctx: tuple[Pipeline, list[InvocationSpec]], req: tuple[EntityId, ItemList]
) -> tuple[EntityId, dict[str, object]]:
    pipeline, invocations = ctx
    user, test_items = req

    result = {}

    _log.debug("running pipeline %s for user %s", pipeline.name, user)
    for inv in invocations:
        inputs = {}
        match inv.items:
            case "test-items":
                inputs["items"] = test_items

        inputs.update(inv.extra_inputs)

        nodes = inv.components.keys()
        outs = pipeline.run_all(*nodes, **inputs)
        for cname, oname in inv.components.items():
            result[oname] = outs[cname]

    return user, result
