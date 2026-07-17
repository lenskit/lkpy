from __future__ import annotations

import math
from collections.abc import Mapping

from optuna import Trial
from pydantic import JsonValue

from lenskit.data import unflatten_dict
from lenskit.logging import get_logger, trace
from lenskit.schemas.tuning import SearchParam, SearchSpace

_log = get_logger(__name__)
type OptunaPoint = dict[str, str | float | int | bool | None]


class SearchPoint:
    """
    Single search point in the Optuna space.
    """

    params: OptunaPoint
    """
    Sampled parameters (Optuna format).
    """

    def __init__(self, params: OptunaPoint):
        self.params = params

    @classmethod
    def defaults(cls, space: SearchSpace, config: dict[str, JsonValue]) -> SearchPoint:
        params = _extract_defaults(space, config)
        return cls(params)

    @classmethod
    def ask(cls, space: SearchSpace, trial: Trial) -> SearchPoint:
        params = _ask_space(trial, space)
        return cls(params)

    def to_config(self) -> dict[str, JsonValue]:
        """
        Get the LensKit configuration for this Optuna search point.
        """
        fixed = {}
        for k, v in self.params.items():
            if k.endswith("%pow2"):
                assert isinstance(v, int)
                fixed[k[: -len("%pow2")]] = 2**v
            else:
                fixed[k] = v
        return unflatten_dict(fixed)


def _ask_space(
    trial: Trial, space: SearchSpace, *, out: OptunaPoint | None = None, prefix: str = ""
) -> OptunaPoint:
    if out is None:
        out = {}

    for name, spec in space.items():
        key = prefix + name
        if isinstance(spec, Mapping):
            _ask_space(trial, spec, out=out, prefix=f"{prefix}{name}.")
        elif spec.type == "int" and spec.scale == "uniform":
            assert isinstance(spec.min, int)
            assert isinstance(spec.max, int)
            out[key] = trial.suggest_int(prefix + name, spec.min, spec.max)
            trace(_log, "sampled int", name=key, min=spec.min, max=spec.max, value=out[key])
        elif spec.type == "int" and spec.scale == "log":
            assert isinstance(spec.min, int)
            assert isinstance(spec.max, int)
            out[key] = trial.suggest_int(prefix + name, spec.min, spec.max, log=True)
            trace(_log, "sampled int/log2", name=key, min=spec.min, max=spec.max, value=out[key])
        elif spec.type == "int" and spec.scale == "pow2":
            assert spec.min is not None
            assert spec.max is not None
            min = int(math.log2(spec.min))
            max = int(math.log2(spec.max))
            key = f"{prefix}{name}%pow2"
            p = trial.suggest_int(key, min, max)
            out[key] = p
            trace(_log, "sampled int/pow2", name=key, min=min, max=max, power=p, value=2**p)
        elif spec.type == "float" and spec.scale == "uniform":
            assert spec.min is not None
            assert spec.max is not None
            out[key] = trial.suggest_float(prefix + name, spec.min, spec.max)
            trace(_log, "sampled float", name=key, min=spec.min, max=spec.max, value=out[key])
        elif spec.type == "float" and spec.scale == "log":
            assert spec.min is not None
            assert spec.max is not None
            out[key] = trial.suggest_float(prefix + name, spec.min, spec.max, log=True)
            trace(_log, "sampled float/log", name=key, min=spec.min, max=spec.max, value=out[key])
        elif spec.type == "bool":
            out[key] = trial.suggest_categorical(prefix + name, [False, True])
            trace(_log, "sampled bool", name=key, value=out[key])
        elif spec.type == "choice":
            out[key] = trial.suggest_categorical(prefix + name, spec.choices)
            trace(_log, "sampled choice", name=key, value=out[key])
        else:  # pragma: nocover
            raise ValueError(f"unsupported configuration {space}")

    return out


def _extract_defaults(
    space: SearchSpace,
    config: Mapping[str, JsonValue],
    *,
    out: OptunaPoint | None = None,
    prefix: str = "",
) -> OptunaPoint:
    # FIXME this whole function is ugly but seems to work
    if out is None:
        out = {}
    for k, v in space.items():
        if isinstance(v, dict):
            _extract_defaults(v, config[k], out=out, prefix=f"{prefix}{k}.")  # type: ignore
        elif isinstance(config, dict):
            assert isinstance(v, SearchParam)
            try:
                out[k] = config[k]
            except KeyError as e:
                if k.endswith("_exp"):
                    out[k] = int(math.log2(config[k[:-4]]))  # type: ignore
                else:
                    raise e
            if v.scale == "pow2":
                out[f"{k}%pow2"] = int(math.log2(out[k]))  # type: ignore
                del out[k]
        else:
            # FIXME this is an ugly hack for single / multiple configs
            out[k] = config
    return out
