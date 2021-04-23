"""
Utilities to manage randomness in LensKit and LensKit experiments.
"""

import warnings
import zlib
import numpy as np
import random
import logging

import seedbank

_log = logging.getLogger(__name__)
derive_seed = seedbank.derive_seed


def get_root_seed():
    """
    Get the root seed.

    Returns:
        numpy.random.SeedSequence: The LensKit root seed.
    """
    warnings.warn('get_root_seed is deprecated, use seedbank.root_seed', DeprecationWarning)
    return seedbank.root_seed()


def init_rng(seed, *keys, propagate=True):
    """
    Initialize the random infrastructure with a seed.  This function should generally be
    called very early in the setup.

    .. warning::

        This method is deprecated. Use :func:`seedbank.initialize` instead.

    Args:
        seed(int or numpy.random.SeedSequence):
            The random seed to initialize with.
        keys:
            Additional keys, to use as a ``spawn_key`` on NumPy 1.17.  Passed to
            :func:`derive_seed`.

        propagate(bool):
            If ``True``, initialize other RNG infrastructure. This currently initializes:

            * :func:`np.random.seed`
            * :func:`random.seed`

            If ``propagate=False``, LensKit is still fully seeded — no component included
            with LensKit uses any of the global RNGs, they all use RNGs seeded with the
            specified seed.

    Returns:
        The random seed.
    """
    warnings.warn('init_rng is deprecated, use seedbank.initialize', DeprecationWarning)
    seedbank.initialize(seed, *keys)


def rng(spec=None, *, legacy=False):
    """
    Get a random number generator.  This is similar to :func:`sklearn.utils.check_random_seed`, but
    it usually returns a :class:`numpy.random.Generator` instead.

    .. warning::

        This method is deprecated. Use :func:`seedbank.numpy_rng` instead.


    Args:
        spec:
            The spec for this RNG.  Can be any of the following types:

            * ``int``
            * ``None``
            * :class:`numpy.random.SeedSequence`
            * :class:`numpy.random.mtrand.RandomState`
            * :class:`numpy.random.Generator`
        legacy(bool):
            If ``True``, return :class:`numpy.random.mtrand.RandomState` instead of a new-style
            :class:`numpy.random.Generator`.

    Returns:
        numpy.random.Generator: A random number generator.
    """
    warnings.warn('rng is deprecated, use seedbank.numpy_rng', DeprecationWarning)

    if legacy:
        return seedbank.numpy_random_state(spec)
    else:
        return seedbank.numpy_rng(spec)


class FixedRNG:
    "RNG provider that always provides the same RNG"
    def __init__(self, rng):
        self.rng = rng

    def __call__(self, *keys):
        return self.rng

    def __str__(self):
        return 'Fixed({})'.format(self.rng)


class DerivingRNG:
    "RNG provider that derives new RNGs from the key"
    def __init__(self, seed, legacy):
        self.seed = seed
        self.legacy = legacy

    def __call__(self, *keys):
        seed = derive_seed(*keys, base=self.seed)
        if self.legacy:
            bg = np.random.MT19937(seed)
            return np.random.RandomState(bg)
        else:
            return np.random.default_rng(seed)

    def __str__(self):
        return 'Derive({})'.format(self.seed)


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

    if spec == 'user':
        return DerivingRNG(derive_seed(), legacy)
    elif isinstance(spec, tuple):
        seed, key = spec
        if key != 'user':
            raise ValueError('unrecognized key %s', key)
        return DerivingRNG(seed, legacy)
    else:
        return FixedRNG(rng(spec, legacy=legacy))
