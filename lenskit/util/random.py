"""
Utilities to manage randomness in LensKit and LensKit experiments.
"""

import zlib
import numpy as np
import random
import warnings
import logging

_log = logging.getLogger(__name__)


class LegacyRNG:
    _seed = None
    _rng = None

    @property
    def seed(self):
        if self._seed is None:
            self._seed = np.random.randint(0, np.iinfo('i4').max)
        return self._seed

    @property
    def int_seed(self):
        return self._seed

    def initialize(self, seed, keys):
        warnings.warn('initializing legacy RNG infrastructure')
        _log.warn('initializing legacy RNG infrastructure - use NumPy 1.17+ for better seeds')
        if keys:
            raise NotImplementedError('legacy RNG does not support seed keys')
        self._seed = seed
        if self._rng is not None:
            del self._rng
        return seed

    def derive(self, base, keys):
        raise NotImplementedError('legacy RNG does not support deriving seeds')

    def rng(self, seed=None):
        if seed is None:
            if self._rng is None:
                seed = self.seed
                self._rng = np.random.RandomState(seed)
            return self._rng
        else:
            return np.random.RandomState(seed)


class ModernRNG:
    _seed = None

    @property
    def seed(self):
        if self._seed is None:
            self._seed = np.random.SeedSequence()
        return self._seed

    @property
    def int_seed(self):
        return self._seed.generate_state(1)[0]

    def initialize(self, seed, keys):
        if isinstance(seed, int):
            seed = np.random.SeedSequence(seed)

        if not isinstance(seed, np.random.SeedSequence):
            raise TypeError('unexpected seed type %s', type(seed))

        if keys:
            seed = self.derive(seed, keys)

        self._seed = seed
        return seed

    def derive(self, base, keys):
        if base is None:
            base = self.seed

        if keys:
            k2 = tuple(_make_int(k) for k in keys)
            return np.random.SeedSequence(base.entropy, spawn_key=base.spawn_key + k2)
        else:
            return base.spawn(1)[0]

    def rng(self, seed=None):
        if seed is None:
            seed, = self.seed.spawn(1)
        elif isinstance(seed, int):
            seed = np.random.SeedSequence(seed)
        return np.random.default_rng(seed)


def get_root_seed():
    """
    Get the root seed.

    Returns:
        numpy.random.SeedSequence: The LensKit root seed.
    """
    return _rng_impl.seed


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
    _rng_impl.initialize(seed, keys)
    _log.info('initialized LensKit RNG with seed %s', _rng_impl.seed)

    if propagate:
        ik = _rng_impl.int_seed
        _log.info('initializing numpy.random and random with seed %u', ik)
        np.random.seed(ik)
        random.seed(ik)
        try:
            import tensorflow as tf
            _tf_seed = getattr(tf.random, 'set_seed', None)
            if _tf_seed is not None:
                _log.debug('setting TensorFlow seed')
                _tf_seed(ik)
            else:
                _log.warn('old TensorFlow, not setting seed')
        except ImportError:
            _log.debug('TensorFlow not available')

    return _rng_impl.seed


def derive_seed(*keys, base=None, none_on_old_numpy=False):
    """
    Derive a seed from the root seed, optionally with additional seed keys.

    Args:
        keys(list of int or str):
            Additional components to add to the spawn key for reproducible derivation.
            If unspecified, the seed's internal counter is incremented (by calling
            :meth:`numpy.random.SeedSequence.spawn`).
        base(numpy.random.SeedSequence):
            The base seed to use.  If ``None``, uses the root seed.
        none_on_old_numpy(bool):
            If ``True``, return ``None`` instead of raising :class:`NotImplementedError`
            if running on an old version of NumPy.
    """
    try:
        return _rng_impl.derive(base, keys)
    except NotImplementedError as e:
        if none_on_old_numpy:
            return None
        else:
            raise e


def rng_seed(spec=None):
    """
    Get a random number generator seed.  ``spec`` is interpreted as in :func:`rng`, and
    is used as follows:

    * If a :class:`numpy.random.SeedSequence`, returned as-is.
    * If an integer, used to create a seed sequence.
    * If ``None``, returns global seed (after initializing).
    * If a :class:`numpy.random.Generator` or :class:`numpy.random.RandomState`, it is
      used to generate an integer that is used to create a seed sequence.

    This function is only available when used with NumPy 1.17 or newer.

    Returns:
        numpy.random.SeedSequence:
            The seed.
    """
    if not hasattr(np.random, 'Generator'):
        raise RuntimeError('rng_seed requires NumPy 1.17')

    if spec is None:
        return _rng_impl.seed
    elif isinstance(spec, int):
        return np.random.SeedSequence(spec)
    elif isinstance(spec, np.random.SeedSequence):
        return spec
    elif hasattr(spec, 'integers'):
        seed = rng.integers(2**32-1)
        return np.random.SeedSequence(seed)
    else:
        raise ValueError('unknown RNG spec ' + str(spec))


def rng(spec=None, *, legacy=False):
    """
    Get a random number generator.  This is similar to :func:`sklearn.utils.check_random_seed`, but
    it usually returns a :class:`numpy.random.Generator` instead.

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

    rng = None
    if isinstance(spec, np.random.RandomState):
        rng = spec
    elif _have_gen and isinstance(spec, np.random.Generator):
        rng = spec
    else:
        rng = _rng_impl.rng(spec)

    if legacy and _have_gen and isinstance(rng, np.random.Generator):
        rng = np.random.RandomState(rng.bit_generator)
    # case where rng is a random state, and we are on new numpy and want a generator, is ok

    return rng


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


# are we on modern NumPy?
_have_gen = hasattr(np.random, 'Generator')
if _have_gen:
    _rng_impl = ModernRNG()
else:
    _rng_impl = LegacyRNG()
