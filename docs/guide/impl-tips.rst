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

LensKit follows `SPEC 7`_ for managing RNG seeds and constructing random number
generation. In general, algorithms using randomization should have an ``rng``
parameter that takes a seed or RNG (of type :py:type:`~lenskit.types.RNGInput`),
and pass this to :py:func:`numpy.random.default_rng` to get a random number
generator. Algorithms that use randomness at predict or recommendation time, not
just training time, should also support the value ``'user'`` for the ``rng``
parameter, and if it is passed, derive a new seed for each user using
:py:func:`lenskit.util.derivable_rng`.

.. _SPEC 7: https://scientific-python.org/specs/spec-0007/
