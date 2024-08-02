# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"Definition of the component interfaces."

# pyright: strict
from __future__ import annotations

from typing_extensions import Any, Generic, Protocol, Self, TypeVar, runtime_checkable

from lenskit.data.dataset import Dataset

# COut is only return, so Component[U] can be assigned to Component[T] if U ≼ T.
COut = TypeVar("COut", covariant=True)


class Component(Protocol, Generic[COut]):
    """
    Interface (protocol) for pipeline components: functions from inputs to outputs.

    Most components will implement additional component protocols, such as:

    * :class:`ConfigurableComponent`
    * :class:`TrainableComponent`

    .. note::

        This protocol is equivalent to ``Callable[..., COut]`` but is defined as a
        protocol so we can define sub-protocols with additional methods.
    """

    def __call__(self, **kwargs: Any) -> COut:
        """
        Apply this component to its input data.

        .. note::

            The protocol definition allows arbitrary keyword arguments (and no
            positional arguments), to work with Python's type system limitations
            and the impossibility of manually writing :class:`~typing.ParamSpec`
            declarations, but component implementations are expected to declare
            specific input arguments with type annotations.
        """
        raise NotImplementedError()


@runtime_checkable
class ConfigurableComponent(Generic[COut], Component[COut], Protocol):
    """
    Interface for configurable pipeline components (those that have
    hyperparameters).  A configurable component supports two additional
    operations:

    * saving its configuration with :meth:`get_config`.
    * creating a new instance from a saved configuration with the class method
      :meth:`from_config`.

    A component must implement both of these methods to be considered
    configurable.

    .. note::

        Configuration data should be JSON-compatible (strings, numbers, etc.).

    .. note::

        This is a subtype of :class:`Component`, so implementations must also
        implement ``__call__`` as specified there.
    """

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> dict[str, object]:
        """
        Reinstantiate this component from configuration values.
        """
        ...

    def get_config(self) -> dict[str, object]:
        """
        Get this component's configured hyperparameters.
        """
        ...


@runtime_checkable
class TrainableComponent(Generic[COut], Component[COut], Protocol):
    """
    Interface for pipeline components that can learn parameters from training
    data, and expose those parameters for serialization as an alternative to
    pickling (components also need to be picklable).

    .. note::

        This is a subtype of :class:`Component`, so implementations must also
        implement ``__call__`` as specified there.
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
