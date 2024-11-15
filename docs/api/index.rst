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
    lenskit.types

.. toctree::
    :caption: Core
    :hidden:

    data
    pipeline
    diagnostics

Components and Models
~~~~~~~~~~~~~~~~~~~~~

These packages provide various recommendation components and models.  Many of
them need to be installed separately.

.. autosummary::
    :toctree: .
    :caption: Components
    :recursive:

    lenskit.algorithms
    lenskit.basic
    lenskit.funksvd
    lenskit.implicit
    lenskit.hpf


Batch Inference and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These package provide evaluation metrics and analysis and batch-inference
support. The evaluation code is not directly linked to the rest of LensKit and
can be used to evaluate the output of any recommender system implementation.

.. autosummary::

    lenskit.metrics

.. toctree::
    :caption: Evaluation
    :hidden:

    metrics

Implementation Helpers
~~~~~~~~~~~~~~~~~~~~~~

These modules provide various utilities and helpers used to implement LensKit,
and may be useful in building new models and components for LensKit.

.. autosummary::
    :toctree: .
    :caption: Implementation Helpers
    :recursive:

    lenskit.math
    lenskit.parallel
    lenskit.util
