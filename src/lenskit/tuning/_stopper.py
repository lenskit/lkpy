# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Any

import numpy as np
import ray.tune

from lenskit.logging import get_logger

_log = get_logger(__name__)


class RelativePlateauStopper(ray.tune.Stopper):
    metric: str
    mode: str
    min_improvement: float
    check_iters: int
    grace_period: int

    results: dict[str, list[float]]

    def __init__(
        self,
        metric: str,
        mode: str,
        min_improvement: float = 0.01,
        check_iters: int = 3,
        grace_period: int = 5,
    ):
        assert check_iters <= grace_period, "check iters cannot be more than grace period"
        self.metric = metric
        self.mode = mode
        self.min_improvement = min_improvement
        self.check_iters = check_iters
        self.grace_period = grace_period
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
        self.results[trial_id] = hist
        if len(hist) < self.grace_period:
            log.debug("within grace period, accepting")
            return False

        hist = np.array(hist)
        # measure relative improvement over previous best entry
        if self.mode == "min":
            best = np.minimum.accumulate(hist)
            imp = hist[1:] / best[:-1] - 1.0
            imp *= -1
        else:
            best = np.maximum.accumulate(hist)
            imp = hist[1:] / best[:-1] - 1.0

        # if we haven't improved enough lately
        if np.all(imp[-self.check_iters :] < self.min_improvement).item():
            log.info(
                "trial plateaued, stopping with last improvement {:.3%}".format(imp[-1]),
                history=hist,
            )
            return True
        else:
            log.debug("continuing with improvement {:.3%}".format(imp[-1]))
            return False

    def stop_all(self) -> bool:
        return False
