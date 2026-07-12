# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Pydantic models for pipeline configuration and serialization support.
"""

# pyright: strict
from __future__ import annotations

import base64
import pickle
import re
from collections import OrderedDict
from hashlib import sha256
from typing import Annotated, Literal, Mapping

from annotated_types import Predicate
from pydantic import AliasChoices, BaseModel, Field, JsonValue, ValidationError
from typing_extensions import Any, Self

from lenskit.diagnostics import PipelineError

VALID_NAME = re.compile(r"^[\w.@%!*?-]+$", re.UNICODE)
UNSET_CODE = "!UNSET"


def check_name(name: str, *, what: str = "node", allow_reserved: bool = False):
    """
    Check that a name is valid.

    Raises:
        ValueError:
            If the specified name is not valid.
    """

    if not VALID_NAME.match(name):
        raise ValueError(f"invalid {what} name “{name}”")
    if name.startswith("_") and not allow_reserved:
        raise ValueError(
            f"invalid {what} name “{name}”, names beginning with “_” are reserved by LensKit"
        )


class PipelineHook(BaseModel):
    """
    A single entry in a pipeline's hook configuration.
    """

    function: str
    priority: Annotated[int, Predicate(lambda x: x != 0)] = 1


class PipelineHooks(BaseModel):
    """
    Hook specifications for a pipeline.
    """

    run: dict[str, list[PipelineHook]] = {}


class PipelineOptions(BaseModel, extra="allow"):
    """
    Options used for pipeline assembly, particularly for extending pipelines.
    """

    base: str | None = None
    "Apply this configuration to a base pipeline."

    default_length: int | None = Field(
        default=None, validation_alias=AliasChoices("default_length", "n")
    )
    """
    For top-*N* base pipelines, the default ranking length to configure.
    """

    fallback_predictor: bool | None = None
    """
    For rating prediction pipelines, enable or disable the default fallback
    predictor.
    """


class PipelineConfigFragment(BaseModel, extra="allow"):
    """
    Configuration fragments that override base configurations.
    """

    options: PipelineOptions | None = None
    "Options for assembling the final pipeline."
    meta: PipelineMeta = Field(default_factory=lambda: PipelineMeta())
    "Pipeline metadata."


class PipelineConfig(BaseModel):
    """
    Root type for serialized pipeline configuration.  A pipeline config contains
    the full configuration, components, and wiring for the pipeline, but does
    not contain the learned parameters.

    Stability:
        Full
    """

    options: PipelineOptions | None = None
    "Options for assembling the final pipeline."
    meta: PipelineMeta = Field(default_factory=lambda: PipelineMeta())
    "Pipeline metadata."
    inputs: list[PipelineInput] = []
    "Pipeline inputs."
    components: OrderedDict[str, PipelineComponent] = Field(default_factory=OrderedDict)
    "Pipeline components, with their configurations and wiring."
    aliases: dict[str, str] = {}
    "Pipeline node aliases."
    default: str | None = None
    "The default node for running this pipeline."
    literals: dict[str, PipelineLiteral] = {}
    "Literals"
    hooks: PipelineHooks = PipelineHooks()
    "The hooks configured for the pipeline."

    def merge_config(self, update: PipelineConfig | PipelineConfigFragment) -> PipelineConfig:
        cfg = self.model_dump()
        patch = update.model_dump(exclude_unset=True, exclude_computed_fields=True)

        merge_configs(cfg, patch)

        return PipelineConfig.model_validate(cfg)

    def merge_component_configs(
        self, configs: Mapping[str, Mapping[str, JsonValue]]
    ) -> PipelineConfig:
        """
        Merge component configuration options into the pipeline configuration.

        This returns a modified copy of the pipeline with the applied
        configurations, and does not modify the configuration in-place.
        """
        pipe = self.model_copy(deep=True)
        for name, cfg in configs.items():
            try:
                comp = pipe.components[name]
            except KeyError:
                raise PipelineError(f"unknown component {name}")  # noqa: B904

            # make sure we have a mutable dictionary
            if comp.config is None:
                base = {}
            else:
                base = dict(comp.config)

            merge_configs(base, cfg)
            comp.config = base

        return pipe


class PipelineMeta(BaseModel):
    """
    Pipeline metadata.

    Stability:
        Full
    """

    name: str | None = None
    "The pipeline name."
    version: str | None = None
    "The pipeline version."
    hash: str | None = None
    """
    The pipeline configuration hash.  This is optional, particularly when
    hand-crafting pipeline configuration files.
    """


class PipelineInput(BaseModel):
    """
    Spcification of a pipeline input.

    Stability:
        Full
    """

    name: str
    "The name for this input."
    types: set[str] | None
    "The list of types for this input."


class PipelineComponent(BaseModel):
    """
    Specification of a pipeline component.
    """

    code: str = Field(validation_alias=AliasChoices("code", "class"), default=UNSET_CODE)
    """
    The path to the component's implementation, either a class or a function.
    This is a Python qualified path of the form ``module:name``.
    """

    config: Mapping[str, JsonValue] | None = None
    """
    The component configuration.  If not provided, the component will be created
    with its default constructor parameters.
    """

    inputs: dict[str, str] = {}
    """
    The component's input wirings, mapping input names to node names.  For
    certain meta-nodes, it is specified as a list instead of a dict.
    """


class PipelineLiteral(BaseModel):
    """
    Literal nodes represented in the pipeline.

    Stability:
        Full
    """

    encoding: Literal["json", "base85"]
    value: JsonValue

    @classmethod
    def represent(cls, data: Any) -> Self:
        try:
            return cls(encoding="json", value=data)
        except ValidationError:
            # data is not basic JSON values, so let's pickle it
            dbytes = pickle.dumps(data)
            return cls(encoding="base85", value=base64.b85encode(dbytes).decode("ascii"))

    def decode(self) -> Any:
        "Decode the represented literal."
        match self.encoding:
            case "json":
                return self.value
            case "base85":
                assert isinstance(self.value, str)
                return pickle.loads(base64.b85decode(self.value))


def hash_config(config: BaseModel) -> str:
    """
    Compute the hash of a configuration model.

    Stability:
        Internal
    """
    json = config.model_dump_json(exclude_none=True)
    h = sha256()
    h.update(json.encode())
    return h.hexdigest()


def merge_configs(
    base: dict[str, Any],
    update: Mapping[str, Any],
    *,
    path: tuple[str, ...] = (),
    _skip_comp: bool = False,
):
    match path, base, update:
        case ("components", _name), _, {**_u} if not _skip_comp:
            # merge component configurations
            assert isinstance(base, dict)
            base_class = base.get("code", None)
            patch_class = update.get("code", UNSET_CODE)
            if patch_class == UNSET_CODE or base_class == patch_class:
                merge_configs(base, update, path=path, _skip_comp=True)
            else:
                merge_configs(
                    base,
                    {k: v for (k, v) in update.items() if k != "config"},
                    path=path,
                    _skip_comp=True,
                )
                base["config"] = update.get("config", {})

        case ("options",), {}, {"base": str(_), **_kws}:
            base.update({k: v for (k, v) in _kws.items()})

        case _:
            for k, v in update.items():
                bv = base.setdefault(k, {})
                match bv, v:
                    case {**_bve}, {**_ve}:
                        merge_configs(bv, v, path=path + (k,))
                    case None, {**_ve}:
                        bv = base[k] = {}
                        merge_configs(bv, v, path=path + (k,))
                    case _:
                        base[k] = v
