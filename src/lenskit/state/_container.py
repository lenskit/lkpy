# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ParameterContainer(Protocol):  # pragma: nocover
    """
    Protocol for components with learned parameters to enable saving, reloading,
    checkpointing, etc.

    Components that learn parameters from training data should implement this
    protocol, and also work when pickled or pickled.  Pickling is sometimes used
    for convenience, but parameter / state dictionaries allow serializing wtih
    tools like ``safetensors`` or ``zarr``.

    Initializing a component with the same configuration as a trained component,
    and loading its parameters with :meth:`load_parameters`, should result in a
    component that is functionally equivalent to the original trained component.

    Stability:
        Experimental
    """

    def get_parameters(self) -> Mapping[str, object]:
        """
        Get the component's parameters.

        Returns:
            The model's parameters, as a dictionary from names to parameter data
            (usually arrays, tensors, etc.).
        """
        raise NotImplementedError()

    def load_parameters(self, state: Mapping[str, Any]) -> None:
        """
        Reload model state from parameters saved via :meth:`get_parameters`.

        Args:
            params:
                The model parameters, as a dictionary from names to parameter
                data (arrays, tensors, etc.), as returned from
                :meth:`get_parameters`.
        """
        raise NotImplementedError()
