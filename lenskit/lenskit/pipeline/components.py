# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"Definition of the component interfaces."

# pyright: strict
from __future__ import annotations

import inspect
from importlib import import_module
from types import FunctionType
from typing import Callable, ClassVar, Generic, TypeAlias

from typing_extensions import Any, Protocol, Self, TypeVar, override, runtime_checkable

from lenskit.data.dataset import Dataset

# COut is only return, so Component[U] can be assigned to Component[T] if U ≼ T.
COut = TypeVar("COut", covariant=True)
PipelineComponent: TypeAlias = Callable[..., COut]


@runtime_checkable
class ConfigurableComponent(Protocol):  # pragma: nocover
    """
    Interface for configurable pipeline components (those that have
    hyperparameters).  A configurable component supports two additional
    operations:

    * saving its configuration with :meth:`get_config`.
    * creating a new instance from a saved configuration with the class method
      :meth:`from_config`.

    A component must implement both of these methods to be considered
    configurable.  For most common cases, extending the :class:`AutoConfig`
    class is sufficient to provide working implementations of these methods.

    If a component is *not* configurable, then it should either be a function or
    a class that can be constructed with no arguments.

    .. note::

        Configuration data should be JSON-compatible (strings, numbers, etc.).

    .. note::

        Implementations must also implement ``__call__``.
    """

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Self:
        """
        Reinstantiate this component from configuration values.
        """
        raise NotImplementedError()

    def get_config(self) -> dict[str, object]:
        """
        Get this component's configured hyperparameters.
        """
        raise NotImplementedError()


@runtime_checkable
class TrainableComponent(Generic[COut], Protocol):  # pragma: nocover
    """
    Interface for pipeline components that can learn parameters from training
    data, and expose those parameters for serialization as an alternative to
    pickling (components also need to be picklable).

    .. note::

        Trainable components must also implement ``__call__``.
    """

    def train(self, data: Dataset) -> Self:
        """
        Train the pipeline component to learn its parameters from a training
        dataset.

        Args:
            data:
                The training dataset.
        Returns:
            The component.
        """
        raise NotImplementedError()

    def get_params(self) -> dict[str, object]:
        """
        Get the model's learned parameters for serialization.

        LensKit components that learn parameters from training data should both
        implement this method and work when pickled and unpickled.  Pickling is
        sometimes used for convenience, but parameter / state dictionaries allow
        serializing wtih tools like ``safetensors``.

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


class AutoConfig(ConfigurableComponent):
    """
    Mixin class providing automatic configuration support based on constructor
    arguments.

    This method provides implementations of :meth:`get_config` and
    :meth:`from_config` that inspect the constructor arguments and instance
    variables to automatically provide configuration support.  By default, all
    constructor parameters will be considered configuration parameters, and
    their values will be read from instance variables of the same name.
    Subclasses can also define :data:`EXTRA_CONFIG_FIELDS` and
    :data:`IGNORED_CONFIG_FIELDS` class variables to modify this behavior.
    Missing attributes are silently ignored.

    In the simple case, you can write a class like this and get config for free:

    .. code:: python

        class MyComponent(AutoConfig):
            some_param: int

            def __init__(self, some_param: int = 20):
                self.some_param = some_param

    For compatibility with pipeline serialization, all configuration data should
    be JSON-compatible.
    """

    EXTRA_CONFIG_FIELDS: ClassVar[list[str]] = []
    """
    Names of instance variables that should be included in the configuration
    dictionary even though they do not correspond to named constructor
    arguments.

    .. note::

        This is rarely needed, and usually needs to be coupled with ``**kwargs``
        in the constructor to make the resulting objects constructible.
    """

    IGNORED_CONFIG_FIELDS: ClassVar[list[str]] = []
    """
    Names of constructor parameters that should be excluded from the
    configuration dictionary.
    """

    @override
    def get_config(self) -> dict[str, object]:
        """
        Get the configuration by inspecting the constructor and instance
        variables.
        """
        sig = inspect.signature(self.__class__)
        names = list(sig.parameters.keys()) + self.EXTRA_CONFIG_FIELDS
        params: dict[str, Any] = {}
        for name in names:
            if name not in self.IGNORED_CONFIG_FIELDS and hasattr(self, name):
                params[name] = getattr(self, name)

        return params

    @override
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Self:
        """
        Create a class from the specified construction.  Configuration elements
        are passed to the constructor as keywrod arguments.
        """
        return cls(**cfg)


def instantiate_component(
    comp: str | type | FunctionType, config: dict[str, Any] | None
) -> Callable[..., object]:
    if isinstance(comp, str):
        mname, oname = comp.split(":", 1)
        mod = import_module(mname)
        comp = getattr(mod, oname)

    # make the type checker happy
    assert not isinstance(comp, str)

    if isinstance(comp, FunctionType):
        return comp
    elif issubclass(comp, ConfigurableComponent):
        if config is None:
            config = {}
        return comp.from_config(config)  # type: ignore
    else:
        return comp()  # type: ignore


class Component(AutoConfig):
    pass
