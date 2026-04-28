# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Implementation of early-stopping rules.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from lenskit.logging import get_logger

_log = get_logger(__name__)


@dataclass(kw_only=True)
class PlateauStopRule:
    """
    General rule to stop training when the metric reaches a relative plateau.
    """

    min_improvement: float = 0.01
    """
    Minimum relative improvement to keep going.
    """

    min_iters: int = 5
    """
    Minimum iterations to run before considering stopping.
    """

    check_iters: int = 3
    """
    Number of iterations to check.  Tuning will stop when none of the most
    recent ``check_iters`` iterations are a sufficient improvement over the best
    so far.

    This is to keep training going if one iteration doesn't improve but the next
    starts improving again.
    """

    mode: Literal["min", "max"]
    """
    Whether we are trying minimize or maximize the metric.
    """

    def should_stop(self, metrics: list[float] | NDArray[np.floating]) -> bool:
        """
        Check if we should stop tuning based on the plateau stopping rules.
        """

        metrics = np.asarray(metrics)

        if len(metrics) < self.min_iters:
            return False

        # accumulate best values so far for each position
        if self.mode == "max":
            best = np.maximum.accumulate(metrics)
            mult = 1
        else:
            best = np.minimum.accumulate(metrics)
            mult = -1

        _log.debug("checking for early stop", n_trials=len(metrics), best=best[-1])

        # compute relative improvement over previous best entry
        imp = metrics[1:] / best[:-1] - 1.0
        # invert for minimizing metrics
        imp *= mult

        # check if we have improved enough lately
        if np.all(imp[-self.check_iters :] < self.min_improvement).item():
            return True
        else:
            return False
