# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
"""
Interfaces and support for model training.
"""

# pyright: strict
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Protocol,
    runtime_checkable,
)

import numpy as np

from lenskit.data.dataset import Dataset
from lenskit.random import RNGInput, random_generator


@dataclass
class TrainingOptions:
    """
    Options and context settings that govern model training.
    """

    retrain: bool = True
    """
    Whether the model should retrain if it is already trained.  If ``False``,
    the model should cleanly skip training if it is already trained.
    """

    device: str | None = None
    """
    The device on which to train (e.g. ``'cuda'``).  May be ignored if the model
    does not support the specified device.
    """

    rng: RNGInput = None
    """
    Random number generator to use for any randomness in the training process.
    This option contains any `SPEC 7`_-compatible random number generator
    specification; the :func:`~lenskit.random.random_generator` will convert
    that into a NumPy :class:`~numpy.random.Generator`.
    """

    _generator: np.random.Generator | None = None

    def random_generator(self) -> np.random.Generator:
        """
        Obtain a random generator from the configured RNG or seed.
        """
        if self._generator is None:
            self._generator = random_generator(self.rng)
        return self._generator


@runtime_checkable
class Trainable(Protocol):  # pragma: nocover
    """
    Interface for components and objects that can learn parameters from training
    data. It supports training and checking if a component has already been
    trained.  The resulting model should be pickleable.  Trainable objects are
    usually also components.

    .. note::

        Trainable components must also implement ``__call__``.

    .. note::

        A future LensKit version will add support for extracting model
        parameters a la Pytorch's ``state_dict``, but this capability was not
        ready for 2025.1.

    Stability:
        Full
    """

    def train(self, data: Dataset, options: TrainingOptions) -> None:
        """
        Train the model to learn its parameters from a training dataset.

        Args:
            data:
                The training dataset.
            options:
                The training options.
        """
        raise NotImplementedError()
