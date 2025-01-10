# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Utilities to manage randomness in LensKit and LensKit experiments.
"""

# pyright: strict
from abc import abstractmethod
from hashlib import md5
from typing import Annotated
from uuid import UUID

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
from pydantic import BeforeValidator, PlainSerializer
from typing_extensions import Any, Literal, Protocol, Sequence, TypeAlias, override

from lenskit.data import RecQuery

SeedLike: TypeAlias = int | np.integer[Any] | Sequence[int] | np.random.SeedSequence
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

SeedDependency = Literal["user"]

_global_rng: Generator | None = None


def validate_seed(seed: Any):
    return seed


def serialize_seed(seed: Any) -> int | Sequence[int] | None:
    if seed is None:
        return None
    elif isinstance(seed, np.random.SeedSequence):
        return seed.entropy
    elif isinstance(seed, Sequence):
        return seed
    else:
        return int(seed)


ConfiguredSeed = Annotated[
    SeedLike | None,
    BeforeValidator(validate_seed, json_schema_input_type=int | Sequence[int] | None),
    PlainSerializer(serialize_seed),
]


def validate_derivable_seed(seed: Any):
    return seed


def serialize_derivable_seed(seed: Any) -> Any:
    if isinstance(seed, tuple):
        seed, dep = seed
        return (serialize_seed(seed), dep)
    elif seed is None:
        return None
    elif isinstance(seed, np.random.SeedSequence):
        return seed.entropy
    elif isinstance(seed, Sequence):
        return seed
    else:
        return int(seed)


DerivableSeed: TypeAlias = Annotated[
    SeedLike | SeedDependency | tuple[SeedLike, SeedDependency] | None,
    BeforeValidator(
        validate_derivable_seed,
        json_schema_input_type=int
        | Sequence[int]
        | None
        | tuple[int | Sequence[int] | None, SeedDependency],
    ),
    PlainSerializer(serialize_derivable_seed),
]


def set_global_rng(seed: RNGInput):
    """
    Set the global default RNG.
    """
    global _global_rng
    _global_rng = default_rng(seed)


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

    - A seed (:type:`~lenskit.util.random.SeedLike`).
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
