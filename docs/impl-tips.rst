Algorithm Implementation Tips
=============================

Implementing algorithms is fun, but there are a few things that are good to keep in mind.

In general, development follows the following:

1. Correct
2. Clear
3. Fast

In that order.  Further, we always want LensKit to be *usable* in an easy fashion.  Code
implementing algorithms, however, may be quite complex in order to achieve good performance.

Random Number Generation
------------------------

LensKit uses :py:mod:`seedbank` for managing RNG seeds and constructing random number generation.

In general, algorithms using randomization should have an ``rng_spec`` parameter that takes a seed
or RNG, and pass this to :py:func:`seedbank.numpy_rng` to get a random number generator. Algorithms
that use randomness at predict or recommendation time, not just training time, should support the
value ``'user'`` for the ``rng`` parameter, and if it is passed, derive a new seed for each user
using :py:func:`seedbank.derive_seed` to allow reproducibility in the face of parallelism for common
experimental designs.  :py:func:`lenskit.util.derivable_rng` automates this logic.
