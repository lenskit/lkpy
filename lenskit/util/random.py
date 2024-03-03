# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Utilities to manage randomness in LensKit and LensKit experiments.
"""

import numpy as np
import seedbank


class FixedRNG:
    "RNG provider that always provides the same RNG"

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, *keys) -> np.random.Generator:
        return self.rng

    def __str__(self):
        return "Fixed({})".format(self.rng)


class DerivingRNG:
    "RNG provider that derives new RNGs from the key"

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, *keys) -> np.random.Generator:
        seed = seedbank.derive_seed(*keys, base=self.seed)
        return seedbank.numpy_rng(seed)

    def __str__(self):
        return "Derive({})".format(self.seed)


def derivable_rng(spec):
    """
    Get a derivable RNG, for use cases where the code needs to be able to reproducibly derive
    sub-RNGs for different keys, such as user IDs.

    Args:
        spec:
            Any value supported by the `seed` parameter of :func:`seedbank.numpy_rng`, in addition
            to the following values:

            * the string ``'user'``
            * a tuple of the form (``seed``, ``'user'``)

            Either of these forms will cause the returned function to re-derive new RNGs.

    Returns:
        function:
            A function taking one (or more) key values, like :func:`derive_seed`, and
            returning a random number generator.
    """

    if spec == "user":
        return DerivingRNG(seedbank.derive_seed())
    elif isinstance(spec, tuple):
        seed, key = spec
        if key != "user":
            raise ValueError("unrecognized key %s", key)
        return DerivingRNG(seed)
    else:
        return FixedRNG(seedbank.numpy_rng(spec))
