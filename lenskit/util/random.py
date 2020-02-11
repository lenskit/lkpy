"""
Utilities to manage randomness in LensKit and LensKit experiments.
"""

import zlib
import numpy as np
import random
import warnings

# are we on modern NumPy?
_have_gen = hasattr(np.random, 'Generator')
# global root seed, on modern NumPy it will be a seed sequence.
_root_seed = None
_global_rng = None


def get_root_seed():
    """
    Get the root seed.

    Returns:
        numpy.random.SeedSequence: The LensKit root seed.
    """
    global _root_seed
    if _root_seed is None:
        if _have_gen:
            _root_seed = np.random.SeedSequence()
        else:
            _root_seed = np.random.randint(0, np.iinfo('i4').max)
    return _root_seed


def _get_rng(legacy=False):
    global _global_rng
    if _have_gen:
        seed = get_root_seed()
        kids = seed.spawn(1)
        if legacy:
            return np.random.RandomState(np.random.MT19937(kids[0]))
        else:
            return np.random.default_rng(kids[0])
    else:
        if _global_rng is None:
            state = get_root_seed()
            _global_rng = np.random.RandomState(state)
        return _global_rng


def _make_int(obj):
    if isinstance(obj, int):
        return obj
    elif isinstance(obj, bytes):
        return zlib.crc32(obj)
    elif isinstance(obj, str):
        return zlib.crc32(obj.encode('utf8'))
    else:
        return ValueError('invalid RNG key ' + str(obj))


def init_rng(seed, *keys, propagate=True):
    """
    Initialize the random infrastructure with a seed.  This function should generally be
    called very early in the setup.

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
    global _root_seed
    if _have_gen:
        if isinstance(seed, int):
            seed = np.random.SeedSequence(seed)
        elif not isinstance(seed, np.random.SeedSequence):
            raise TypeError('seed must be an int or SeedSequence, got {}'.format(type(seed)))
        _root_seed = seed
        if keys:
            _root_seed = derive_seed(*keys, base=_root_seed)

        if propagate:
            nps, pys = seed.spawn(2)
            np.random.seed(nps.generate_state(1))
            random.seed(pys.generate_state(1)[0])

        return _root_seed

    else:
        warnings.warn('initializing random seeds with legacy infrastructure')
        _root_seed = seed
        if propagate:
            np.random.seed(seed)
            random.seed(seed)
        return seed


def derive_seed(*keys, base=None):
    """
    Derive a seed from the root seed, optionally with additional seed keys.

    Args:
        keys(list of int or str):
            Additional components to add to the spawn key for reproducible derivation.
            If unspecified, the seed's internal counter is incremented.
        base(numpy.random.SeedSequence):
            The base seed to use.  If ``None``, uses the root seed.
    """
    if not _have_gen:
        raise NotImplementedError('derive_seed requires NumPy 1.17 or newer')

    if base is None:
        base = get_root_seed()

    if keys:
        k2 = tuple(_make_int(k) for k in keys)
        return np.random.SeedSequence(base.entropy, spawn_key=base.spawn_key + k2)
    else:
        return base.spawn(1)[0]


def rng(seed=None, *, legacy=False):
    """
    Get a random number generator.  This is similar to :func:`sklearn.utils.check_random_seed`, but
    it usually returns a :class:`numpy.random.Generator` instead.

    Args:
        seed:
            The seed for this RNG.  Can be any of the following types:

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

    if _have_gen and legacy and isinstance(seed, np.random.Generator):
        # convert a new generator to a NumPy random state
        return np.random.RandomState(seed.bit_generator)
    elif seed is None:
        return _get_rng(legacy)
    elif isinstance(seed, int):
        if legacy:
            return np.random.RandomState(seed)
        else:
            return np.random.default_rng(seed)
    elif _have_gen and isinstance(seed, np.random.SeedSequence):
        if legacy:
            return np.random.RandomState(np.random.MT19937(seed))
        else:
            return np.random.default_rng(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    elif _have_gen and isinstance(seed, np.random.Generator):
        return seed
    else:
        raise ValueError('invalid random seed ' + str(seed))


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
            Any value supported by the `seed` parameter of :func:`rng`, in addition to the
            following values:

            * the string ``'user'``
            * a tuple of the form (`seed`, ``'user'``)

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
