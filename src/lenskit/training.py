# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Interfaces and support for model training.
"""

# pyright: strict
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Protocol, overload, override, runtime_checkable

import numpy as np

from lenskit.data import Dataset
from lenskit.logging import get_logger, item_progress
from lenskit.pipeline.components import Component
from lenskit.random import RNGInput, random_generator

if TYPE_CHECKING:
    import torch

_log = get_logger(__name__)


@dataclass(frozen=True)
class TrainingOptions:
    """
    Options and context settings that govern model training.
    """

    retrain: bool = True
    """
    Whether the model should retrain if it is already trained.  If ``False``,
    the component is allowed to skip training if it is already trained.

    In the common case of training pipelines, this flag is examined by
    :meth:`lenskit.pipeline.Pipeline.train`: if it is ``False``, that method
    skips training any components that are already trained.  Custom training
    code that wishes to avoid retraining models should check
    :meth:`Trainable.is_trained` instead of assuming that individual components
    will respect this flag.

    .. note::

        This division of responsibility is to reduce the need for repetitive
        code: since implementing components seems to be a more common activity
        than logic that directly trains components (as opposed to pipelines) in
        ordinary LensKit use, making training code responsible for skipping
        retrain instead of requiring that of every component implementation
        allows individual implementations to be slightly simpler, without
        requiring separate options classes for pipeline and component training.

    .. versionchanged:: 2026.1

        Added the :meth:`is_trained` method that implementers must now also
        provide.
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

    environment: dict[str, str] = field(default_factory=lambda: {})
    """
    Additional training environment variables to control training behavior.
    Variables and their meanings are defined by individual components.
    Variables in this option override system environment variables when fetched
    with :meth:`envvar`.
    """

    torch_profiler: torch.profiler.profile | None = None
    """
    Torch profiler for profiling training options.
    """

    def step_profiler(self):
        """
        Signal to active profiler(s) that a new step has completed.
        """
        if self.torch_profiler is not None:
            self.torch_profiler.step()

    @overload
    def random_generator(self, *, type: Literal["numpy"] = "numpy") -> np.random.Generator: ...
    @overload
    def random_generator(self, *, type: Literal["torch"]) -> torch.Generator: ...
    def random_generator(
        self, *, type: Literal["numpy", "torch"] = "numpy"
    ) -> np.random.Generator | torch.Generator:
        """
        Obtain a random generator from the configured RNG or seed.

        .. note::

            Each call to this method will return a fresh generator from the same
            seed.  Components should call it once at the beginning of their
            training procesess.
        """
        return random_generator(self.rng, type=type)

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

    @overload
    def env_var(self, name: str, default: str) -> str: ...
    @overload
    def env_var(self, name: str, default: str | None = None) -> str | None: ...
    def env_var(self, name: str, default: str | None = None) -> str | None:
        """
        Fetch a training environment variable.  Variables are first looked up in
        :attr:`environment`, then in :attr:`os.environ`.

        .. seealso::
            :attr:`environment`, :ref:`training-config`

        Args:
            name:
                The full name of the environment variable.
            default:
                Default value to return if the environment varible is not specified.
        Returns:
            The environment variable's value, or ``default``.
        """
        if name in self.environment:
            return self.environment[name]
        else:
            return os.environ.get(name, default)

    def env_flag(self, name: str, *, default: bool = False) -> bool:
        """
        Query a boolean flag from the environment.
        """
        val = self.env_var(name)
        if val is None:
            return default
        elif isinstance(val, bool):
            return val
        elif isinstance(val, int):
            return bool(val)
        elif re.match(r"^\d+$", val):
            return bool(int(val))
        elif re.match(r"^(?:t(?:rue)?|y(?:es)?)$", val, re.IGNORECASE):
            return True
        elif re.match(r"^(?:f(?:alse)?|n(?:o)?)$", val, re.IGNORECASE):
            return False
        else:
            raise ValueError(f"unrecognized boolean value {name}={val}")


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

    ..
        -   They should also usually implement
            :class:`~lenskit.state.ParameterContainer`, to allow the learned
            parameters to be serialized and deserialized without pickling.

    Stability:
        Full
    """

    def is_trained(self) -> bool:  # pragma: nocover
        """
        Query if this component has already been trained.
        """
        raise NotImplementedError()

    def train(self, data: Dataset, options: TrainingOptions) -> None:  # pragma: nocover
        """
        Train the model to learn its parameters from a training dataset.

        Args:
            data:
                The training dataset.
            options:
                The training options.
        """
        raise NotImplementedError()


class UsesTrainer(Component, ABC, Trainable):
    """
    Base class for models that implement :class:`Trainable` via a
    :class:`ModelTrainer`.

    The component's configuration must have an ``epochs`` attribute defining the
    number of epochs to train.

    Stability:
        Full
    """

    trained_epochs: int = 0

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

    @override
    def is_trained(self):
        return self.trained_epochs > 0

    @override
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

    ..

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

    def finalize(self) -> None:
        """
        Finish the training process, cleaning up any unneeded data structures
        and doing any finalization steps to the model.

        The default implementation does nothing.
        """
        pass
