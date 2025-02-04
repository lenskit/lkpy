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

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from lenskit.data.dataset import Dataset
from lenskit.logging import get_logger, item_progress
from lenskit.random import RNGInput, random_generator

_log = get_logger(__name__)


@dataclass(frozen=True)
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

    def random_generator(self) -> np.random.Generator:
        """
        Obtain a random generator from the configured RNG or seed.

        .. note::

            Each call to this method will return a fresh generator from the same
            seed.  Components should call it once at the beginning of their
            training procesess.
        """
        return random_generator(self.rng)

    def configured_device(self, *, gpu_default: bool = False) -> str:
        """
        Get the configured device, consulting environment variables and defaults
        if necessary.  It looks for a device in the following order:

        1. The :attr:`device`, if specified on this object.
        2. The :envvar:`LK_DEVICE` environment variable.
        3. If CUDA is enabled and ``gpu_default`` is ``True``, return `"cuda"`
        4. The CPU.

        Args:
            gpu_default:
                Whether a CUDA GPU should be preferred if it is available and no
                device has been specified.
        """
        import torch

        if self.device is not None:
            return self.device
        elif dev := os.environ.get("LK_DEVICE", None):
            return dev
        elif gpu_default and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"


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


class IterativeTraining(ABC, Trainable):
    """
    Base class for components that support iterative training.  This both
    automates the :meth:`Trainable.train` method for iterative training in terms
    of initialization, epoch, and finalization methods, and exposes those
    methods to client code that may wish to directly control the iterative
    training process.

    Stability:
        Full
    """

    trained_epochs: int = 0
    """
    The number of epochs for which this model has been trained.
    """

    @property
    def expected_training_epochs(self) -> int | None:
        """
        Get the number of training epochs expected to run.  The default
        implementation looks for an ``epochs`` attribute on the configuration
        object (``self.config``).
        """
        cfg = getattr(self, "config", None)
        if cfg:
            return getattr(cfg, "epochs", None)

    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()) -> None:
        """
        Implementation of :meth:`Trainable.train` that uses the training loop.
        It also uses the :attr:`trained_epochs` attribute to detect if the model
        has already been trained for the purposes of honoring
        :attr:`TrainingOptions.retrain`, and updates that attribute as model
        training progresses.
        """
        if self.trained_epochs > 0 and not options.retrain:
            return

        self.trained_epochs = 0
        log = _log.bind(model=f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        log.info("training model")
        n = self.expected_training_epochs
        log.debug("creating training loop")
        loop = self.training_loop(data, options)
        log.debug("beginning training iterations")
        with item_progress("Training iterations", total=n) as pb:
            for i, metrics in enumerate(loop, 1):
                metrics = metrics or {}
                log.info("finished epoch", epoch=i, **metrics)
                self.trained_epochs += 1
                pb.update()

        log.info("model training finished", epochs=self.trained_epochs)

    @abstractmethod
    def training_loop(
        self, data: Dataset, options: TrainingOptions
    ) -> Iterator[dict[str, float] | None]:
        """
        Training loop implementation, to be supplied by the derived class.  This
        method should return a iterator that, when iterated, will perform each
        training epoch; when training is complete, it should finalize the model
        and signal iteration completion.

        Each epoch can yield metrics, such as training or validation loss, to be
        logged with structured logging and can be used by calling code to do
        other analysis.

        See :ref:`iterative-training` for more details on writing iterative
        training loops.
        """
        raise NotImplementedError()
