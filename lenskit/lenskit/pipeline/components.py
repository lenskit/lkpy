# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"Definition of the component interfaces."

# pyright: strict
from __future__ import annotations

import inspect
import json
import warnings
from abc import abstractmethod
from importlib import import_module
from types import FunctionType
from typing import (
    Any,
    Callable,
    Mapping,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    get_origin,
    runtime_checkable,
)

from pydantic import JsonValue, TypeAdapter

from lenskit.data.dataset import Dataset

from .types import Lazy

P = ParamSpec("P")
T = TypeVar("T")
Cfg = TypeVar("Cfg")
# COut is only return, so Component[U] can be assigned to Component[T] if U ≼ T.
COut = TypeVar("COut", covariant=True)
PipelineFunction: TypeAlias = Callable[..., COut]


@runtime_checkable
class Trainable(Protocol):  # pragma: nocover
    """
    Interface for components that can learn parameters from training data. It
    supports trainingand checking if a component has already been trained.
    Trained components need to be picklable.

    .. note::

        Trainable components must also implement ``__call__``.

    .. note::

        A future LensKit version will add support for extracting model
        parameters a la Pytorch's ``state_dict``, but this capability was not
        ready for 2025.1.

    Stability:
        Full
    """

    @property
    def is_trained(self) -> bool:
        """
        Check if this model has already been trained.
        """
        raise NotImplementedError()

    def train(self, data: Dataset) -> None:
        """
        Train the pipeline component to learn its parameters from a training
        dataset.

        Args:
            data:
                The training dataset.
            retrain:
                If ``True``, retrain the model even if it has already been
                trained.
        """
        raise NotImplementedError()


@runtime_checkable
class ParameterContainer(Protocol):  # pragma: nocover
    """
    Future protocol for components with learned parameters.

    .. important::

        This protocol is not yet used.

    Stability:
        Experimental
    """

    def get_params(self) -> dict[str, object]:
        """
        Protocol for components with parameters that can be extracted, saved,
        and re-loaded.

        LensKit components that learn parameters from training data should both
        implement this method and work when pickled and unpickled.  Pickling is
        sometimes used for convenience, but parameter / state dictionaries allow
        serializing wtih tools like ``safetensors``.

        .. todo::

            This protocol is not yet used for anything.

        Args:
            include_caches:
                Whether the parameter dictionary should include ephemeral
                caching structures only used for runtime performance
                optimizations.

        Returns:
            The model's parameters, as a dictionary from names to parameter data
            (usually arrays, tensors, etc.).
        """
        raise NotImplementedError()

    def load_params(self, params: dict[str, object]) -> None:
        """
        Reload model state from parameters saved via :meth:`get_params`.
        """
        raise NotImplementedError()


class Component:
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

    config: Any
    """
    The component configuration object.  Component classes that support
    configuration **must** redefine this attribute with their specific
    configuration class type, which can be a Python dataclass or a Pydantic
    model class.
    """

    def __init_subclass__(cls, **kwargs: Any):
        super().__init__(**kwargs)
        cctype = cls._config_class()
        if not cctype:
            warnings.warn("component class {} does not define a config attribute".format(cctype))

    def __init__(self, config: object | None = None, **kwargs: Any):
        if config is None:
            config = self.validate_config(kwargs)
        elif kwargs:
            raise RuntimeError("cannot supply both a configuration object and kwargs")

        self.config = config

    @classmethod
    def _config_class(cls) -> type | None:
        annots = inspect.get_annotations(cls, eval_str=True)
        ct = annots.get("config", None)
        if ct is None or ct == Any:
            return None

        if isinstance(ct, type):
            return ct
        else:
            warnings.warn("config attribute is not annotated with a plain type")
            return get_origin(ct)

    def dump_config(self) -> dict[str, JsonValue]:
        """
        Dump the configuration to JSON-serializable format.
        """
        cfg_cls = self._config_class()
        if cfg_cls:
            return TypeAdapter(cfg_cls).dump_python(self.config, mode="json")
        else:
            return {}

    @classmethod
    def validate_config(cls, data: Mapping[str, JsonValue] | None = None) -> object | None:
        """
        Validate and return a configuration object for this component.
        """
        if data is None:
            data = {}
        cfg_cls = cls._config_class()
        if cfg_cls:
            return TypeAdapter(cfg_cls).validate_python(data)
        elif data:
            raise RuntimeError(
                "supplied configuration options but {} has no config class".format(cls.__name__)
            )
        else:
            return None

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """
        Run the pipeline's operation and produce a result.  This is the key
        method for components to implement.

        .. note::

            Due to limitations of Python's type model, derived classes will have
            a type error overriding this method when using strict type checking,
            because it is very cumbersome (if not impossible) to propagate
            parameter names and types through to a base class.  The solution is
            to either use basic type checking for implementations, or to disable
            the typechecker on the ``__call__`` signature definition.
        """
        ...

    def __repr__(self) -> str:
        params = json.dumps(self.dump_config(), indent=4)
        return f"<{self.__class__.__name__} {params}>"


def instantiate_component(
    comp: str | type | FunctionType, config: dict[str, Any] | None
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
        return comp(cfg)
    else:
        return comp()  # type: ignore


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
