# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Utilities to manage randomness in LensKit and LensKit experiments.
"""

# pyright: strict
from __future__ import annotations

import os
from abc import abstractmethod
from hashlib import md5
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
from typing_extensions import Any, Literal, Protocol, Sequence, TypeAlias, override

if TYPE_CHECKING:  # avoid circular import
    from lenskit.data import RecQuery

SeedLike: TypeAlias = int | Sequence[int] | np.random.SeedSequence
"""
Type for RNG seeds (see `SPEC 7`_).

.. _SPEC 7: https://scientific-python.org/specs/spec-0007/
"""

RNGLike: TypeAlias = np.random.Generator | np.random.BitGenerator
"""
Type for random number generators as inputs (see `SPEC 7`_).

.. _SPEC 7: https://scientific-python.org/specs/spec-0007/
"""

RNGInput: TypeAlias = SeedLike | RNGLike | None
"""
Type for RNG inputs (see `SPEC 7`_).

.. _SPEC 7: https://scientific-python.org/specs/spec-0007/
"""

ConfiguredSeed: TypeAlias = int | Sequence[int] | None
"""
Random number seed that can be configured.
"""

SeedDependency = Literal["user"]

_global_seed: SeedSequence | None = None
_global_rng: Generator | None = None


DerivableSeed: TypeAlias = ConfiguredSeed | SeedDependency | tuple[ConfiguredSeed, SeedDependency]


def load_seed(file: Path | os.PathLike[str] | str, key: str = "random.seed") -> SeedSequence:
    """
    Load a seed from a configuration file.

    Args:
        file:
            The path to the configuration file.
        key:
            The path within the configuration object to the random seed.
    """
    data: Any
    file = Path(file)

    with file.open("rb") as f:
        match file.suffix:
            case ".toml":
                import tomllib

                data = tomllib.load(f)
            case ".yaml":
                import yaml

                data = yaml.load(f, yaml.SafeLoader)
            case ".json":
                import json

                data = json.load(f)
            case _:
                raise ValueError(f"unsupported config file {file.name}")

    parts = key.split(".")
    for part in parts:
        if isinstance(data, dict):
            data = data[part]
        else:
            raise TypeError(f"unsupported data type {type(data)}")

    return make_seed(data)


def set_global_rng(seed: RNGInput):
    """
    Set the global default RNG.

    .. deprecated: 2025.3.0
        Deprecated alias for :func:`init_global_rng`.
    """
    init_global_rng(seed)


def init_global_rng(
    seed: RNGInput, *, seed_stdlib: bool = True, seed_numpy: bool = True, seed_pytorch: bool = True
):
    """
    Set the global default RNG.

    Args:
        seed:
            The seed to set.
        seed_stdlib:
            If ``True``, also seed the Python standard library RNG.
        seed_numpy:
            If ``True``, also seed the deprecated NumPy global RNG.
        seed_torch:
            If ``True``, also seed PyTorch.
    """
    global _global_rng, _global_seed

    if isinstance(seed, RNGLike):
        _global_rng = default_rng(seed)
        int_seed = _global_rng.integers(np.iinfo("i4").max)
    else:
        _global_seed = make_seed(seed)
        int_seed = _global_seed.generate_state(1)[0]
        _global_rng = default_rng(_global_seed)

    if seed_stdlib:
        import random

        random.seed(int(int_seed))

    if seed_numpy:
        np.random.seed(int_seed)

    if seed_pytorch:
        import torch

        torch.manual_seed(int_seed)  # type: ignore


def random_generator(seed: RNGInput = None) -> Generator:
    """
    Create a a random generator with the given seed, falling back to a global
    generator if no seed is provided.  If no global generator has been
    configured with :func:`set_global_rng`, it returns a fresh random RNG.
    """

    global _global_rng
    if seed is None and _global_rng is not None:
        return _global_rng
    else:
        return default_rng(seed)


def make_seed(
    *keys: SeedSequence | int | str | bytes | UUID | Sequence[int] | np.integer[Any] | None,
) -> SeedSequence:
    """
    Make an RNG seed from an input key, allowing strings as seed material.
    """
    seed: list[int] = []
    for key in keys:
        if key is None:
            continue
        elif isinstance(key, SeedSequence):
            ent = key.entropy
            if ent is None:
                continue
            elif isinstance(ent, int):
                seed.append(ent)
            else:
                seed += ent
        elif isinstance(key, np.integer):
            seed.append(key.item())
        elif isinstance(key, int):
            seed.append(key)
        elif isinstance(key, UUID):
            seed.append(_bytes_seed(key.bytes))
        elif isinstance(key, str):
            seed.append(_bytes_seed(key.encode("utf8")))
        elif isinstance(key, bytes):
            seed.append(_bytes_seed(key))
        elif isinstance(key, Sequence):  # type: ignore
            seed += key
        else:  # pragma: nocover
            raise TypeError(f"invalid key input: {key}")

    return SeedSequence(seed)


def _bytes_seed(key: bytes) -> int:
    digest = md5(key).digest()
    arr = np.frombuffer(digest, np.int32)
    return abs(np.bitwise_xor.reduce(arr).item())


class RNGFactory(Protocol):
    """
    Protocol for RNG factories that can do dynamic (e.g. per-user) seeding.
    """

    @abstractmethod
    def __call__(self, query: RecQuery | None) -> Generator:
        raise NotImplementedError()


class FixedRNG(RNGFactory):
    "RNG provider that always provides the same RNG"

    rng: Generator

    def __init__(self, rng: Generator):
        self.rng = rng

    @override
    def __call__(self, query: RecQuery | None = None) -> Generator:
        return self.rng

    def __str__(self):
        return "Fixed({})".format(self.rng)


class DerivingRNG(RNGFactory):
    "RNG provider that derives new RNGs from the key"

    seed: np.random.SeedSequence

    def __init__(self, seed: np.random.SeedSequence):
        self.seed = seed

    @override
    def __call__(self, query: RecQuery | None = None) -> Generator:
        if query is None or query.user_id is None:
            return np.random.default_rng(self.seed.spawn(1)[0])
        else:
            seed = make_seed(self.seed, query.user_id)  # type: ignore
            return default_rng(seed)

    def __str__(self):
        return "Derive({})".format(self.seed)


def derivable_rng(spec: DerivableSeed) -> RNGFactory:
    """
    RNGs that may be derivable from data in the query. These are for designs
    that need to be able to reproducibly derive RNGs for different keys, like
    user IDs (to make a “random” recommender produce the same sequence for the
    same user).

    Seed specifications may be any of the following:

    - A seed (:type:`~lenskit.random.SeedLike`).
    - The value ``'user'``, which will derive a seed from the query user ID.
    - A tuple of the form ``(seed, 'user')``, that will use ``seed`` as the
      basis and drive from it a new seed based on the user ID.

    .. seealso:: :ref:`rng`

    Args:
        spec:
            The seed specification.

    Returns:
        function:
            A function taking one (or more) key values, like
            :func:`derive_seed`, and returning a random number generator.
    """

    if spec == "user":
        return DerivingRNG(SeedSequence())
    elif isinstance(spec, tuple):
        seed, key = spec
        if key != "user":
            raise ValueError("unrecognized key %s", key)
        return DerivingRNG(make_seed(seed))
    else:
        return FixedRNG(default_rng(spec))
