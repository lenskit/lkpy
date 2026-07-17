from collections.abc import Iterable
from dataclasses import dataclass
from typing import override

import numpy as np
from optuna import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState
from pydantic import JsonValue

from lenskit.schemas.tuning import TuningSpec

from .._base import TuneResults
from .point import SearchPoint


@dataclass
class OptunaTuneResults(TuneResults):
    spec: TuningSpec
    study: Study
    iterative: bool

    def num_trials(self) -> int:
        return len([t for t in self.study.trials if t.state != TrialState.FAIL])

    def num_failed(self) -> int:
        return len([t for t in self.study.trials if t.state == TrialState.FAIL])

    @override
    def trials(self) -> Iterable[dict[str, JsonValue]]:
        for trial in self.study.trials:
            if trial.state != TrialState.FAIL:
                yield self._trial_result(trial)

    @override
    def epochs(self) -> Iterable[dict[str, JsonValue]]:
        if not self.iterative:
            return []

        for n, trial in enumerate(self.study.trials):
            if trial.state == TrialState.FAIL:
                continue

            for i in range(trial.last_step + 1):
                out_row = {
                    "trial": n,
                    "epoch": i,
                    self.spec.search.metric: trial.intermediate_values.get(i),
                }
                yield out_row

    def best_config(self):
        best = self.study.best_trial
        point = SearchPoint(best.params)
        if self.iterative:
            vals = [best.intermediate_values.get(i) for i in range(best.last_step + 1)]
            match self.study.direction:
                case StudyDirection.MAXIMIZE:
                    point.params["epochs"] = np.argmax(vals).item() + 1
                case StudyDirection.MINIMIZE:
                    point.params["epochs"] = np.argmin(vals).item() + 1
                case _:  # pragma: nocover
                    raise RuntimeError("unexpected study direction")

        return point.to_config()

    def best_result(self) -> dict[str, JsonValue]:
        return self._trial_result(self.study.best_trial)

    def _trial_result(
        self, trial: FrozenTrial, *, include_config: bool = True
    ) -> dict[str, JsonValue]:
        assert self.spec.search.metric is not None
        result: dict[str, JsonValue]

        if self.iterative:
            assert trial.last_step is not None
            vals = [trial.intermediate_values.get(i) for i in range(trial.last_step + 1)]
            epoch = np.argmax(vals).item()
            result = {
                self.spec.search.metric: vals[epoch],
                "epochs": epoch,
                "attempted_epochs": len(trial.intermediate_values),
                "final_metric": trial.value,
            }
        else:
            result = {self.spec.search.metric: trial.value}

        if include_config:
            result["params"] = trial.params
            result["config"] = self._trial_config(trial)
        return result

    def _trial_config(self, trial: FrozenTrial) -> dict[str, JsonValue]:
        point = SearchPoint(trial.params)
        if self.iterative:
            assert trial.last_step is not None
            vals = [trial.intermediate_values.get(i) for i in range(trial.last_step + 1)]
            point.params["epochs"] = np.argmax(vals).item()

        return point.to_config()
