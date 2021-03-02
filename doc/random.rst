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
It uses the new :py:class:`numpy.random.Generator` and
:py:class:`numpy.random.SeedSequence` facilities to provide consistent random
number generation and initialization.

.. note::
   For fully reproducible research, including random seeds and the use thereof,
   make sure that you are running on the same platform with the same verions of all
   packages (particularly LensKit, NumPy, SciPy, Pandas, and related packages).

Developers *using* LensKit will be primarily intrested in the :py:func:`init_rng`
function, so they can initialize LensKit's random seed.  LensKit components using
randomization also take an ``rng`` option, usually in their constructor, to set
the seed on a per-operation basis; if the script is straightforward and performs
LensKit operations in a deterministic order (e.g. does not train multiple models
in parallel), initializing the global RNG is sufficient.

Developers writing new LensKit algorithms that use randomization will also need
pay attention to the :py:func:`rng` function, along with :py:func:`derivable_rng`
and :py:func:`derive_seed` if predictions or recommendations, not just model
training, requires random values.  Their constructors should take a parameter
``rng_spec`` to specify the RNG initialization.

Seeds
-----

LensKit random number generation starts from a global root seed, accessible with
:py:func:`get_root_seed`. This seed can be initialized with :py:func:`init_rng`.

.. autofunction:: init_rng

.. autofunction:: derive_seed

.. autofunction:: get_root_seed

Random Number Generators
------------------------

These functions create actual RNGs from the LensKit global seed or a user-provided
seed. They can produce both new-style :py:class:`numpy.random.Generator` RNGs and 
legacy :py:class:`numpy.random.mtrand.RandomState`; the latter is needed because
some libraries, such as Pandas and scikit-learn, do not yet know what to do with
a new-style RNG.

.. autofunction:: rng

.. autofunction:: derivable_rng
