Configuring Components
======================

Each LensKit component has its own configuration, specified through constructor
parameters and documented in the component class's API documentation. New
components in your code or in add-on packages are similarly configured, and
extending :class:`~lenskit.pipeline.Component` is sufficient to get reasonable
support from the pipeline's configuration serialization capabilities in most
cases.

LensKit components follow a few conventions for their configuration, documented
here.

List Length
~~~~~~~~~~~

Ranking and selection components typically provide two ways to specify the
desired list length: a configuration option (constructor parameter) and a
runtime parameter (input), both named ``n`` and type ``int | None``.  This
allows list length to be baked into a pipeline configuration, and also allows
that length to be specified or overridden at runtime.  If both lengths are
specified, the runtime length takes precedence.

.. _config-rng:

Random Seeds
~~~~~~~~~~~~

.. _SPEC 7: https://scientific-python.org/specs/spec-0007/

LensKit components follow `SPEC 7`_ for specifying random number seeds.
Components that use randomization (either at runtime, or to set initial
conditions for training) have a constructor parameter `rng` that takes either a
:class:`~numpy.random.Generator` or seed material.  If you want reproducible
stochastic pipelines, configure the random seeds for your components.

.. important::

    If you specify random seeds, we strongly recommend specifying seeds instead of
    generators, so that the seed can be included in serialized configurations.

.. versionchanged:: 2025.1

    Now that `SPEC 7`_ has standardized RNG seeding across the scientific Python
    ecosystem, we use that with some lightweight helpers in the
    :mod:`lenskit.util.random` module instead of using SeedBank.

Derived Seeds
-------------

Recommendation provides a particular challenge for deterministic random behavior
in the face of multiple recommendation requests, particularly when those
requests are parallelized so their arrival order is nondeterministic.

To handle this, LensKit components that randomize responses at runtime (such as
:class:`lenskit.basic.RandomSelector` and :class:`lenskit.basic.SoftmaxRanker`)
support *derivable RNGs*.

.. seealso:: :func:`lenskit.util.derivable_rng`
