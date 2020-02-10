Random Number Generation
========================

.. py:module:: lenskit.util.random

Current best practice for reproducible science in machine learning — including, 
but not limited to, recommender systems — is to use fixed random seeds so results
can be reproduced precisely.  This is useful both for reproducing the results
themselves and for debugging.

To test for seed sensitivity, the entire experiment can be re-run with a different
random seed and the conclusions compared.

LensKit is built to support this experimental design, making consistent use of
configurable random number generators throughout its algorithm implementations.
When run against NumPy 1.17 or later, it uses the new :py:class:`numpy.random.Generator`
and :py:class:`numpy.random.SeedSequence` facilities to provide consistent random
number generation and initialization.  LensKit is compatible with older versions
of NumPy, but the RNG reproducibility logic will not fully function, and some
functions will not work.

Seeds
-----

LensKit random number generation starts from a global seed, stored in :py:data:`global_seed`.
This seed can be initialized with :py:func:`init_rng`.


.. py:data:: global_seed
   
   The global seed for LensKit RNGs.

.. autofunction:: init_rng

.. autofunction:: derive_seed

Random Number Generators
------------------------

All LensKit components that use randomization take an ``rng`` parameter, usually to their constructor.
This parameter is passed to the :py:func:`rng` function to obtain a random number generator that is
compatible with the code's requirements.

This function can produce both new-style :py:class:`numpy.random.Generator` RNGs and legacy
:py:class:`numpy.random.mtrand.RandomState`; the latter is needed because some libraries, such as Pandas and
scikit-learn, do not yet know what to do with a new-style RNG.

.. autofunction:: rng

.. autofunction:: derivable_rng
