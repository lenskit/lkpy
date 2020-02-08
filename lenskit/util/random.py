"""
Utilities to manage randomness in LensKit and LensKit experiments.
"""

import numpy as np

_have_gen = hasattr(np.random, 'Generator')
_seed = None
_generator = None


def _get_rng():
    global _seed
    if _have_gen:
        if _seed is None:
            _seed = np.random.SeedSequence()
        kids = _seed.spawn(1)
        return np.random.default_rng(kids[0])
    else:
        return np.random.mtrand._rand


def rng(seed=None, *, legacy=False):
    """
    Get a random number generator.  This is similar to :func:`sklearn.utils.check_random_seed`, but
    it usually returns a :cls:`numpy.random.Generator` instead.

    Args:
        seed(int or None or np.random.RandomState or np.random.Generator):
            The seed for this RNG.
        legacy(bool):
            If ``True``, return :cls:`numpy.random.RandomState` instead of a new-style
            :cls:`numpy.random.Generator`.
    """

    if _have_gen and legacy and isinstance(seed, np.random.Generator):
        # convert a new generator to a NumPy random state
        return np.random.RandomState(seed.bit_generator)
    elif seed is None:
        if legacy:
            return np.random.mtrand._rand
        else:
            return _get_rng()
    elif isinstance(seed, int):
        if legacy:
            return np.random.RandomState(seed)
        else:
            return np.random.default_rng(seed)
    elif _have_gen and isinstance(seed, np.random.SeedSequence):
        if legacy:
            return np.random.RandomState(seed.generate_state(1))
        else:
            return np.random.default_rng(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    elif _have_gen and isinstance(seed, np.random.Generator):
        return seed
    else:
        raise ValueError('invalid random seed ' + str(seed))
