# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"Definition of the component interfaces."

# pyright: strict
from __future__ import annotations

from typing import Callable, TypeAlias

from typing_extensions import Any, Generic, Protocol, Self, TypeVar, runtime_checkable

from lenskit.data.dataset import Dataset

# COut is only return, so Component[U] can be assigned to Component[T] if U ≼ T.
COut = TypeVar("COut", covariant=True)
Component: TypeAlias = Callable[..., COut]


@runtime_checkable
class ConfigurableComponent(Generic[COut], Protocol):
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

        Implementations must also implement ``__call__``.
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
class TrainableComponent(Generic[COut], Protocol):
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
