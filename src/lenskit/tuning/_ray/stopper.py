# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Any, Literal

import ray.tune

from lenskit.logging import get_logger

from .._stopping import PlateauStopRule

_log = get_logger(__name__)


class RelativePlateauStopper(ray.tune.Stopper):
    metric: str
    rule: PlateauStopRule

    results: dict[str, list[float]]

    def __init__(
        self,
        metric: str,
        mode: Literal["min", "max"],
        min_improvement: float = 0.01,
        check_iters: int = 3,
        grace_period: int = 5,
    ):
        assert check_iters <= grace_period, "check iters cannot be more than grace period"
        self.metric = metric
        self.rule = PlateauStopRule(
            mode=mode,
            min_improvement=min_improvement,
            min_iters=grace_period,
            check_iters=check_iters,
        )

        self.results = {}

    def __call__(self, trial_id: str, result: dict[str, Any]) -> bool:
        epoch = result["training_iteration"]
        mr = result[self.metric]
        log = _log.bind(trial=trial_id, epoch=epoch, **{self.metric: mr})

        hist = self.results.get(trial_id, [])
        if len(hist) >= result["training_iteration"]:
            log.debug("truncating history", len=len(hist))
            hist = hist[: result["training_iteration"] - 1]
        hist.append(mr)

        if self.rule.should_stop(hist):
            log.info("trial plateaued", history=hist)
            return True
        else:
            log.debug("continuing", history=hist)
            return False

    def stop_all(self) -> bool:
        return False
