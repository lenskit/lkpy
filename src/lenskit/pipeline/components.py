# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"Definition of the component interfaces."

# pyright: strict
from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from importlib import import_module
from inspect import isabstract, signature
from types import FunctionType, NoneType

from pydantic import JsonValue, TypeAdapter
from typing_extensions import (
    Any,
    Callable,
    Generic,
    Mapping,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from .types import Lazy, TypecheckWarning

P = ParamSpec("P")
T = TypeVar("T")
CFG = TypeVar("CFG")
CArgs = ParamSpec("CArgs", default=...)
"""
Argument type for a component.  It is difficult to actually specify this, but
using this default parameter spec allows :class:`Component` subclasses to
typecheck by declaring the base class :meth:`~Component.__call__` to have
unknown parameters.
"""
# COut is only return, so Component[U] can be assigned to Component[T] if U ≼ T.
COut = TypeVar("COut", covariant=True, default=Any)
"""
Return type for a component.
"""
PipelineFunction: TypeAlias = Callable[..., COut]
"""
Pure-function interface for pipeline functions.
"""


@runtime_checkable
class ComponentConstructor(Protocol, Generic[CFG, COut]):
    """
    Protocol for component constructors.
    """

    def __call__(self, config: CFG | None = None) -> Component[COut]: ...

    def config_class(self) -> type[CFG] | None: ...

    def validate_config(self, data: Any = None) -> CFG | None: ...


class Component(ABC, Generic[COut, CArgs]):
    """
    Base class for pipeline component objects.  Any component that is not just a
    function should extend this class.

    Pipeline components support configuration (e.g., hyperparameters or random
    seeds) through Pydantic models or Python dataclasses; see
    :ref:`component-config` for further details.  If the pipeline's
    configuration class is ``C``, it has the following:

    1. The configuration is exposed through an instance variable ``config``.
    2. The constructor accepts the configuration object as its first parameter,
       also named ``config``, and saves this in the member variable.

    The base class constructor handles both of these, so long as you declare the
    type of the ``config`` member::

        class MyComponent(Component):
            config: MyComponentConfig

            ...

    If you do not declare a ``config`` attribute, the base class will assume the
    pipeline uses no configuration.

    To work as components, derived classes also need to implement a ``__call__``
    method to perform their operations.

    Args:
        config:
            The configuration object.  If ``None``, the configuration class will
            be instantiated with ``kwargs``.

    Stability:
        Full
    """

    config: Any = None
    """
    The component configuration object.  Component classes that support
    configuration **must** redefine this attribute with their specific
    configuration class type, which can be a Python dataclass or a Pydantic
    model class.
    """

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            try:
                ct = cls.config_class(return_any=True)
            except NameError:
                # can't look up the config class, so we can't check it
                pass
            else:
                if ct == Any:
                    warnings.warn(
                        "component class {} does not define a config attribute type".format(
                            cls.__qualname__
                        ),
                        stacklevel=2,
                    )

    def __init__(self, config: object | None = None, **kwargs: Any):
        if config is None:
            config = self.validate_config(kwargs)
        elif kwargs:
            raise RuntimeError("cannot supply both a configuration object and kwargs")

        cfg_cls = self.config_class(return_any=True)
        if cfg_cls == Any:
            warnings.warn(
                "component class {} does not define a config attribute type".format(
                    self.__class__.__qualname__
                ),
                stacklevel=2,
            )
        elif cfg_cls and not isinstance(config, cfg_cls):
            raise TypeError(f"invalid configuration type {type(config)}")

        self.config = config

    @classmethod
    def config_class(cls, return_any: bool = False) -> type | None:
        hints = get_type_hints(cls)
        ct = hints.get("config", None)
        if ct == NoneType:
            return None
        elif ct is None or ct == Any:
            if return_any:
                return ct
            else:
                return None
        elif isinstance(ct, type):
            return ct
        else:
            warnings.warn("config attribute is not annotated with a plain type", stacklevel=2)
            return get_origin(ct)

    def dump_config(self) -> dict[str, JsonValue]:
        """
        Dump the configuration to JSON-serializable format.
        """
        cfg_cls = self.config_class()
        if cfg_cls:
            return TypeAdapter(cfg_cls).dump_python(self.config, mode="json")  # type: ignore
        else:
            return {}

    @classmethod
    def validate_config(cls, data: Mapping[str, JsonValue] | None = None) -> object | None:
        """
        Validate and return a configuration object for this component.
        """
        if data is None:
            data = {}
        cfg_cls = cls.config_class()
        if cfg_cls:
            return TypeAdapter(cfg_cls).validate_python(data)  # type: ignore
        elif data:  # pragma: nocover
            raise RuntimeError(
                "supplied configuration options but {} has no config class".format(cls.__name__)
            )
        else:
            return None

    @abstractmethod
    def __call__(self, *args: CArgs.args, **kwargs: CArgs.kwargs) -> COut:  # pragma: nocover
        """
        Run the pipeline's operation and produce a result.  This is the key
        method for components to implement.
        """
        ...

    def __repr__(self) -> str:
        params = json.dumps(self.dump_config(), indent=4)
        return f"<{self.__class__.__name__} {params}>"


def instantiate_component(
    comp: str | type | FunctionType, config: Mapping[str, Any] | None
) -> Callable[..., object]:
    """
    Utility function to instantiate a component given its class, function, or
    string representation.

    Stability:
        Internal
    """
    if isinstance(comp, str):
        mname, oname = comp.split(":", 1)
        mod = import_module(mname)
        comp = getattr(mod, oname)

    # make the type checker happy
    assert not isinstance(comp, str)

    if isinstance(comp, FunctionType):
        return comp
    elif issubclass(comp, Component):
        cfg = comp.validate_config(config)
        return comp(cfg)  # type: ignore
    else:  # pragma: nocover
        return comp()  # type: ignore


def component_inputs(
    component: Component[COut] | ComponentConstructor[Any, COut] | PipelineFunction[COut],
    *,
    warn_on_missing: bool = True,
) -> dict[str, type | None]:
    if isinstance(component, FunctionType):
        function = component
    elif hasattr(component, "__call__"):
        function = getattr(component, "__call__")
    else:
        raise TypeError("invalid component " + repr(component))

    types = get_type_hints(function)
    sig = signature(function)

    inputs: dict[str, type | None] = {}
    for param in sig.parameters.values():
        if param.name == "self":
            continue

        if pt := types.get(param.name, None):
            inputs[param.name] = pt
        else:
            if warn_on_missing:
                warnings.warn(
                    f"parameter {param.name} of component {component} has no type annotation",
                    TypecheckWarning,
                    2,
                )
            inputs[param.name] = None

    return inputs


def component_return_type(
    component: Component[COut] | ComponentConstructor[Any, COut] | PipelineFunction[COut],
) -> type | None:
    if isinstance(component, FunctionType):
        function = component
    elif hasattr(component, "__call__"):
        function = getattr(component, "__call__")
    else:
        raise TypeError("invalid component " + repr(component))

    types = get_type_hints(function)
    return types.get("return", None)


def fallback_on_none(primary: T | None, fallback: Lazy[T]) -> T:
    """
    Fallback to a second component if the primary input is `None`.

    Stability:
        Caller
    """
    if primary is not None:
        return primary
    else:
        return fallback.get()
