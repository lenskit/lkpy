# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"Definition of the component interfaces."

# pyright: strict
from __future__ import annotations

import json
from abc import abstractmethod
from importlib import import_module
from types import FunctionType
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Mapping,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from pydantic import JsonValue, TypeAdapter

from lenskit.data.dataset import Dataset

from .types import Lazy

P = ParamSpec("P")
T = TypeVar("T")
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


class Component(Generic[COut]):
    """
    Base class for pipeline component objects.  Any component that is not just a
    function should extend this class.

    Pipeline components support configuration (e.g., hyperparameters or random
    seeds) through Pydantic models or Python dataclasses; see
    :ref:`component-config` for further details.  If the pipeline's
    configuration class is ``C``, it has the following:

    1. The class variable CONFIG_CLASS stores C.
    2. The configuration is exposed through an instance variable ``config``.
    3. The constructor accepts the configuration object as its first parameter,
       also named ``config``, and saves this in the member variable.

    The base class constructor handles 2 and 3, and can handle 1 if you pass the
    configuration class as a ``config`` arugment in the class definition::

        class MyComponent(Component, config_class=MyComponentConfig):
            pass

    If the pipeline uses no configuration, then ``CONFIG_CLASS`` should be
    ``None`` (or the component should be a plain function instead).  This is the
    default if no configuration is specified.

    To work as components, derived classes also need to implement a ``__call__``
    method to perform their operations.

    Args:
        config:
            The configuration object.  If ``None``, the configuration class will
            be instantiated with ``kwargs``.

    Stability:
        Full
    """

    CONFIG_CLASS: ClassVar[type[object] | None] = None
    config: object | None

    def __init_subclass__(cls, /, config_class: type[object] | None = None, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if config_class is not None:
            cls.CONFIG_CLASS = config_class  # type: ignore

    def __init__(self, config: object | None = None, **kwargs: Any):
        if config is None:
            config = self.validate_config(kwargs)
        elif kwargs:
            raise RuntimeError("cannot supply both a configuration object and kwargs")

        self.config = config

    def dump_config(self) -> dict[str, JsonValue]:
        """
        Dump the configuration to JSON-serializable format.
        """
        if self.CONFIG_CLASS:
            return TypeAdapter(self.CONFIG_CLASS).dump_python(self.config, mode="json")
        else:
            return {}

    @classmethod
    def validate_config(cls, data: Mapping[str, JsonValue] | None = None) -> object | None:
        """
        Validate and return a configuration object for this component.
        """
        if data is None:
            data = {}
        if cls.CONFIG_CLASS:
            return TypeAdapter(cls.CONFIG_CLASS).validate_python(data)
        elif data:
            raise RuntimeError(
                "supplied configuration options but {} has no config class".format(cls.__name__)
            )
        else:
            return None

    @abstractmethod
    def __call__(self, **kwargs: Any) -> COut:
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
        params = json.dumps(self.get_config(), indent=2)
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
        if comp.CONFIG_CLASS is not None:
            return comp(TypeAdapter(comp.CONFIG_CLASS).validate_python(config))
        else:
            return comp()
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
