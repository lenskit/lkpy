.. _conventions:

Component Conventions
=====================

The components shipped with LensKit follow certain conventions to make their
configuration and operation consistent and predictable. We encourage you to
follow these conventions in your own code as well.

List Length
~~~~~~~~~~~

Ranking and selection components typically provide two ways to specify the
desired list length: a configuration option (constructor parameter) and a
runtime parameter (input), both named ``n`` and type ``int | None``.  This
allows list length to be baked into a pipeline configuration, and also allows
that length to be specified or overridden at runtime.  If both lengths are
specified, the runtime length takes precedence.

.. _rng:

Random Seeds
~~~~~~~~~~~~

.. _SPEC 7: https://scientific-python.org/specs/spec-0007/

LensKit components follow `SPEC 7`_ for specifying random number seeds.  If you
want reproducible stochastic pipelines, configure the random seeds for your
components and/or training process.

Components that use randomization at **inference time** take either seed
material or a :class:`~numpy.random.Generator` as an ``rng`` constructor
parameter; if seed material is supplied, that seed should be considered part of
the configuration (see the source code in :mod:`lenskit.basic.random` for
examples).

Components that use randomization at **training time** should obtain their
generator or seed from the :attr:`~lenskit.training.TrainingOptions`.  This
makes it easy to configure a seed for the training process without needing
to configure each component.

Other LensKit code, such as the `data splitting support <./splitting>`_, follow
SPEC 7 directly by accepting an ``rng`` keyword parameter.

.. important::

    If you specify random seeds for component configurations, we strongly
    recommend specifying seeds instead of generators, so that the seed can be
    included in serialized configurations.

.. versionchanged:: 2025.1

    Now that `SPEC 7`_ has standardized RNG seeding across the scientific Python
    ecosystem, we use that with some lightweight helpers in the
    :mod:`lenskit.random` module instead of using SeedBank.

LensKit extends SPEC 7 with a global RNG that components can use as a fallback,
to make it easier to configure system-wide generation for things like tests.
This is configured with :func:`~lenskit.random.set_global_rng`. Components can
use the :func:`~lenskit.random_generator` function to convert seed material or a
generator into a NumPy generator, falling back to the global RNG if one is
specified.

Derived Seeds
-------------

Recommendation provides a particular challenge for deterministic random behavior
in the face of multiple recommendation requests, particularly when those
requests are parallelized, resulting in nondeterministic arrival orders.

To handle this, LensKit components that randomize responses at runtime (such as
:class:`~lenskit.basic.RandomSelector` and :class:`~lenskit.basic.SoftmaxRanker`)
support *derivable RNGs*.  They are selected by passing the string ``'user'`` as
the RNG seed, or a tuple of the form ``(seed, 'user')``.  When configured with
such a seed, the component will deterministically derive a seed for each request
based on the request's userID.  This means that, for the same set of items and
starting seed (and LensKit, NumPy, etc. versions),
:class:`~lenskit.basic.RandomSelector` will return the *same* items for a given
user, and different items for other users, regardless of the order in which
those users are processed.

.. seealso:: :func:`lenskit.util.derivable_rng`
