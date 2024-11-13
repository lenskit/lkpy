# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from lenskit.data.items import ItemList
from lenskit.pipeline import Pipeline

ItemSource: TypeAlias = None | Literal["test-items"]
"""
Types of items that can be returned.
"""


@dataclass
class InvocationSpec:
    """
    Specification for a single pipeline invocation, to record one or more
    pipeine component outputs for a test user.
    """

    name: str
    "A name for this invocation."
    components: dict[str, str]
    "The components to measure and return, mapped to their output names."
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
    """

    pipeline: Pipeline
    invocations: list[InvocationSpec]

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
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

    def run(self, n_jobs: int | None = None) -> dict[str, ItemList]:
        pass
