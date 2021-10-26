Algorithm Implementation Tips
=============================

Implementing algorithms is fun, but there are a few things that are good to keep in mind.

In general, development follows the following:

1. Correct
2. Clear
3. Fast

In that order.  Further, we always want LensKit to be *usable* in an easy fashion.  Code
implementing algorithms, however, may be quite complex in order to achieve good performance.

Performance
-----------

.. _CSR: https://csr.lenskit.org

We use Numba to optimize critical code paths and provide parallelism in a number of cases,
such as ALS training.  See the ALS source code for examples.

We also use the CSR_ package for sparse matrices that are usable from Numba-accelerated code,
and to provide unified access to important sparse matrix operations that use MKL acceleration
when available.  Previous versions of LensKit included the MKL code directly, but we have
moved that logic over into CSR.

If you are working on an algorithm implementation that needs access to additional MKL operations,
please add the relevant operations to CSR to keep LensKit pure Python + Numba.  We do not have
plans to re-add the MKL wrapper logic to the LensKit core.

Pickling and Sharing
--------------------

.. _binpickle: https://binpickle.lenskit.org

LensKit uses binpickle_ quite a bit to save and reload models and to share model data between
concurrent processes.  This generally just works, and you don't need to implement any particular
save/load logic in order to have your algorithm be savable and sharable.

There are a few exceptions, though.

**If your algorithm updates state after fitting**, this should *not* be pickled.  An example of
this would be caching predictions or recommendations to save time in subsequent calls.  Only the
model parameters and estimated parameters should be pickled.  If you have caches or other
ephemeral structures, override ``__getstate__`` and ``__setstate__`` to exclude them from the
saved data and to initialize caches to empty values on unpickling.

**If your model excludes secondary data structures from pickling**, such as a reverse index of
user-item interactions, then you should only exclude them when pickling for serialization. When
pickling for model sharing (see :py:func:`lenskit.sharing.in_share_context`), you should include
the derived structures so they can also be shared.

**If your algorithm uses subsidiary models as a part of the training process**, but does not need them
for prediction or recommendation, then consider overriding ``__getstate__`` to remove the underlying
model or replace it with a cloned copy (with :py:func:`lenskit.util.clone`) to reduce serialized
disk space (and deserialized memory use).

Random Number Generation
------------------------

LensKit uses :py:mod:`seedbank` for managing RNG seeds and constructing random number generation.

In general, algorithms using randomization should have an ``rng_spec`` parameter that takes a seed
or RNG, and pass this to :py:func:`seedbank.numpy_rng` to get a random number generator. Algorithms
that use randomness at predict or recommendation time, not just training time, should support the
value ``'user'`` for the ``rng`` parameter, and if it is passed, derive a new seed for each user
using :py:func:`seedbank.derive_seed` to allow reproducibility in the face of parallelism for common
experimental designs.  :py:func:`lenskit.util.derivable_rng` automates this logic.
