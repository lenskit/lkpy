API Reference
=============

These pages contain the reference documentation for the LensKit modules,
classes, etc.

Core Abstractions
~~~~~~~~~~~~~~~~~

.. autosummary::
    lenskit.data
    lenskit.pipeline
    lenskit.diagnostics
    lenskit.operations
    lenskit.training
    lenskit.state
    lenskit.config

.. toctree::
    :caption: Core
    :hidden:

    data
    pipeline
    operations
    diagnostics
    training
    state
    config

Components and Models
~~~~~~~~~~~~~~~~~~~~~

These packages provide various recommendation components and models.  Many of
them need to be installed separately.

.. autosummary::
    :toctree: .
    :caption: Components
    :recursive:

    lenskit.basic
    lenskit.stochastic
    lenskit.knn
    lenskit.als
    lenskit.flexmf
    lenskit.sklearn
    lenskit.funksvd
    lenskit.implicit
    lenskit.hpf
    lenskit.graphs


Batch Inference and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These package provide evaluation metrics and analysis and batch-inference
support. The evaluation code is not directly linked to the rest of LensKit and
can be used to evaluate the output of any recommender system implementation.

.. autosummary::

    lenskit.batch
    lenskit.metrics
    lenskit.splitting
    lenskit.tuning

.. toctree::
    :caption: Evaluation
    :hidden:

    batch
    metrics
    splitting
    lenskit.tuning

Implementation Helpers
~~~~~~~~~~~~~~~~~~~~~~

These modules provide various utilities and helpers used to implement LensKit,
and may be useful in building new models and components for LensKit.

.. autosummary::
    :toctree: .
    :caption: Implementation Helpers
    :recursive:

    lenskit.logging
    lenskit.math
    lenskit.parallel
    lenskit.random
    lenskit.stats
    lenskit.testing
    lenskit.torch
    lenskit._accel
