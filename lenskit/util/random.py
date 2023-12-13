"""
Utilities to manage randomness in LensKit and LensKit experiments.
"""

import numpy as np

import seedbank


class FixedRNG:
    "RNG provider that always provides the same RNG"

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, *keys):
        return self.rng

    def __str__(self):
        return "Fixed({})".format(self.rng)


class DerivingRNG:
    "RNG provider that derives new RNGs from the key"

    def __init__(self, seed, legacy):
        self.seed = seed
        self.legacy = legacy

    def __call__(self, *keys):
        seed = seedbank.derive_seed(*keys, base=self.seed)
        if self.legacy:
            bg = np.random.MT19937(seed)
            return np.random.RandomState(bg)
        else:
            return np.random.default_rng(seed)

    def __str__(self):
        return "Derive({})".format(self.seed)


def derivable_rng(spec, *, legacy=False):
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
            returning a random number generator (the type of which is determined by
            the ``legacy`` parameter).
    """

    if spec == "user":
        return DerivingRNG(seedbank.derive_seed(), legacy)
    elif isinstance(spec, tuple):
        seed, key = spec
        if key != "user":
            raise ValueError("unrecognized key %s", key)
        return DerivingRNG(seed, legacy)
    else:
        return FixedRNG(seedbank.numpy_rng(spec, legacy=legacy))
