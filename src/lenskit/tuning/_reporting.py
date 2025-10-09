# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Any

import numpy as np
import ray.tune
import ray.tune.result

from lenskit.logging import Progress, get_logger, item_progress

_log = get_logger("codex.tuning")


@dataclass
class TrialProgress:
    bar: Any
    count: int


class StatusCallback(ray.tune.Callback):
    def __init__(self, model: str | None, ds: str | None):
        self.log = _log.bind(model=model, dataset=ds)

    def on_trial_result(self, iteration, trials, trial, result, **info):
        metrics = {n: v for (n, v) in result.items() if n in ["RBP", "NDCG", "RecipRank", "RMSE"]}
        self.log.info("new trial result", iter=iteration, id=trial.trial_id, **metrics)


class ProgressReport(ray.tune.ProgressReporter):
    job_name: str | None = None
    metric = None
    mode = None
    best_metric = None
    _bar: Progress
    _task_bars: dict[str, TrialProgress]

    def __init__(self, name: str | None = None):
        super().__init__()
        self.job_name = name
        self.done = set()

    def setup(self, start_time=None, total_samples=None, metric=None, mode=None, **kwargs):
        super().setup(start_time, total_samples, metric, mode, **kwargs)

        _log.info("setting up tuning status", total_samples=total_samples, metric=metric, mode=mode)
        extra = {metric: ".3f"} if metric is not None else {}
        label = "Tuning " + (self.job_name or "trials")
        self._bar = item_progress(label, total_samples, extra)
        self._task_bars = {}
        self.metric = metric
        self.mode = mode

    def report(self, trials, done, *sys_info):
        _log.debug("reporting trial completion", trial_count=len(trials))

        if done:
            _log.debug("search complete", trial_count=len(trials))
            for bar in self._task_bars.values():
                bar.bar.finish()
            self._bar.finish()
        else:
            total = len(trials)
            if total <= self._bar.total:
                total = None

            n_new = 0
            for trial in trials:
                self._update_metric(trial)
                if trial.status == "TERMINATED" and trial.trial_id not in self.done:
                    self.done.add(trial.trial_id)
                    n_new += 1
                    _log.debug("finished trial", id=trial.trial_id, config=trial.config)

                if trial.status != "RUNNING" and trial.trial_id in self._task_bars:
                    self._task_bars[trial.trial_id].bar.finish()
                    del self._task_bars[trial.trial_id]

                if (
                    trial.status == "RUNNING"
                    and trial.last_result
                    and "training_iteration" in trial.last_result
                ):
                    tp = self._task_bars.get(trial.trial_id, None)
                    epoch = trial.last_result["training_iteration"]
                    t_total = trial.last_result.get("max_epochs", None)
                    if tp is None:
                        bar = item_progress(
                            "Trial " + trial.trial_id,
                            total=t_total,
                            fields={self.metric: ".3f"},
                        )
                        tp = TrialProgress(bar, 0)
                        self._task_bars[trial.trial_id] = tp

                    if epoch > tp.count:
                        tp.bar.update(
                            epoch - tp.count,
                            total=t_total,
                            **{self.metric: trial.last_result[self.metric]},
                        )
                        tp.count = epoch

            extra = {self.metric: self.best_metric or np.nan}
            self._bar.update(n_new, total=total, **extra)

    def should_report(self, trials, done=False):
        return True

    def _update_metric(self, trial):
        if self.metric is not None and trial.last_result and self.metric in trial.last_result:
            mv = trial.last_result[self.metric]
            if self.best_metric is None:
                self.best_metric = mv
                return True
            elif self.mode == "max" and mv > self.best_metric:
                self.best_metric = mv
                return True
            elif self.mode == "min" and mv < self.best_metric:
                self.best_metric = mv
                return True

        return False
