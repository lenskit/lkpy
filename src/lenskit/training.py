# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Interfaces and support for model training.
"""

# pyright: strict
from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

import numpy as np

from lenskit.data.dataset import Dataset
from lenskit.logging import get_logger, item_progress
from lenskit.pipeline.components import Component
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
    trained.  This protocol only captures the concept of trainability; most
    trainable components should have other properties and behaviors as well:

    -   They are usually components (:class:`~lenskit.pipeline.Component`), with
        an appropriate ``__call__`` method.
    -   They should be pickleable.
    -   They should also usually implement
        :class:`~lenskit.state.ParameterContainer`, to allow the learned
        parameters to be serialized and deserialized without pickling.

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

    .. deprecated: 2025.3

        This base class is deprecated in favor of :class:`ModelTrainer` and
        :class:`UsesTrainer`, which are a better fit for integration with
        tools like Ray Tune.

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
        log.debug("beginning training epochs")
        with item_progress("Training epochs", total=n) as pb:
            start = perf_counter()
            for i, metrics in enumerate(loop, 1):
                metrics = metrics or {}
                now = perf_counter()
                elapsed = now - start
                log.info("finished epoch", time="{:.1f}s".format(elapsed), epoch=i, **metrics)
                self.trained_epochs += 1
                start = now
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


class UsesTrainer(IterativeTraining, Component, ABC):
    """
    Base class for models that implement :class:`Trainable` via a
    :class:`ModelTrainer`.  This class implements :class:`IterativeTraining` for
    compatibility, but the :class:`IterativeTraining` interface is deprecated.

    The component's configuration must have an ``epochs`` attribute noting the
    number of epochs to train.

    Stability:
        Full
    """

    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()) -> None:
        """
        Implementation of :meth:`Trainable.train` that uses the model trainer.
        """
        if self.trained_epochs > 0 and not options.retrain:
            return

        self.trained_epochs = 0

        log = _log.bind(model=f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        log.info("training model")
        n = self.expected_training_epochs
        assert n is not None, "no training epochs configured"
        log.debug("creating model trainer")
        trainer = self.create_trainer(data, options)

        log.debug("beginning training epochs")
        with item_progress("Training epochs", total=n) as pb:
            start = perf_counter()
            for i in range(1, n + 1):
                metrics = trainer.train_epoch()
                metrics = metrics or {}
                now = perf_counter()
                elapsed = now - start
                log.info("finished epoch", time="{:.1f}s".format(elapsed), epoch=i, **metrics)
                self.trained_epochs += 1
                start = now
                pb.update()

        log.debug("finalizing model")
        trainer.finalize()

        log.info("model training finished", epochs=self.trained_epochs)

    def training_loop(self, data: Dataset, options: TrainingOptions):
        warnings.warn("IteratativeTraining API is deprecated", DeprecationWarning)

        log = _log.bind(model=f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        log.info("training model")
        n = self.expected_training_epochs
        assert n is not None, "no training epochs configured"
        log.debug("creating model trainer")
        trainer = self.create_trainer(data, options)

        log.debug("beginning training epochs")
        with item_progress("Training epochs", total=n) as pb:
            start = perf_counter()
            for i in range(1, n + 1):
                metrics = trainer.train_epoch()
                metrics = metrics or {}
                now = perf_counter()
                elapsed = now - start
                log.info("finished epoch", time="{:.1f}s".format(elapsed), epoch=i, **metrics)
                start = now
                self.trained_epochs += 1
                pb.update()
                yield metrics

        log.debug("finalizing model")
        trainer.finalize()

    @abstractmethod
    def create_trainer(
        self, data: Dataset, options: TrainingOptions
    ) -> ModelTrainer:  # pragma: nocover
        """
        Create a model trainer to train this model.
        """


class ModelTrainer(ABC):
    """
    Protocol implemented by iterative trainers for models.  Models that
    implement :class:`UsesTrainer` will return an object implementing this
    protocol from their :meth:`~UsesTrainer.create_trainer` method.

    This protocol only defines the core aspects of training a model. Trainers
    should also implement :class:`~lenskit.state.ParameterContainer` to allow
    training to be checkpointed and resumed.

    It is also a good idea for the trainer to be pickleable, but the parameter
    container interface is the primary mechanism for checkpointing.

    Stability:
        Full
    """

    @abstractmethod
    def train_epoch(self) -> dict[str, float] | None:
        """
        Perform one epoch of the training process, optionally returning metrics
        on the training behavior.  After each training iteration, the mmodel
        must be usable.
        """

    @abstractmethod
    def finalize(self) -> None:
        """
        Finish the training process, cleaning up any unneeded data structures
        and doing any finalization steps to the model.
        """
