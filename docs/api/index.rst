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
    lenskit.types

.. toctree::
    :caption: Core
    :hidden:

    data
    pipeline
    operations
    diagnostics

Components and Models
~~~~~~~~~~~~~~~~~~~~~

These packages provide various recommendation components and models.  Many of
them need to be installed separately.

.. autosummary::
    :toctree: .
    :caption: Components
    :recursive:

    lenskit.basic
    lenskit.knn
    lenskit.als
    lenskit.sklearn
    lenskit.funksvd
    lenskit.implicit
    lenskit.hpf


Batch Inference and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These package provide evaluation metrics and analysis and batch-inference
support. The evaluation code is not directly linked to the rest of LensKit and
can be used to evaluate the output of any recommender system implementation.

.. autosummary::

    lenskit.batch
    lenskit.metrics
    lenskit.splitting

.. toctree::
    :caption: Evaluation
    :hidden:

    batch
    metrics
    splitting

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
    lenskit.util
