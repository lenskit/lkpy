API Reference
=============

These pages contain the reference documentation for the LensKit modules,
classes, etc.

Core Abstractions
~~~~~~~~~~~~~~~~~

.. autoapisummary::

    lenskit
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

    lenskit/index
    lenskit/data/index
    lenskit/pipeline/index
    lenskit/operations/index
    lenskit/diagnostics/index
    lenskit/training/index
    lenskit/state/index
    lenskit/config/index

Components and Models
~~~~~~~~~~~~~~~~~~~~~

These packages provide various recommendation components and models provided
as first-party implementations by LensKit.

.. autoapisummary::
    :caption: Components

    lenskit.als
    lenskit.basic
    lenskit.flexmf
    lenskit.funksvd
    lenskit.knn
    lenskit.sklearn
    lenskit.stochastic

.. toctree::
    :caption: Components
    :hidden:

    lenskit/als/index
    lenskit/basic/index
    lenskit/flexmf/index
    lenskit/funksvd/index
    lenskit/knn/index
    lenskit/sklearn/index
    lenskit/stochastic/index

Batch Inference and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These package provide evaluation metrics and analysis and batch-inference
support. The evaluation code is not directly linked to the rest of LensKit and
can be used to evaluate the output of any recommender system implementation.

.. autoapisummary::

    lenskit.batch
    lenskit.metrics
    lenskit.splitting
    lenskit.tuning

.. toctree::
    :caption: Evaluation
    :hidden:

    lenskit/batch/index
    lenskit/metrics/index
    lenskit/splitting/index
    lenskit/tuning/index

Add-On Components
~~~~~~~~~~~~~~~~~

These packages contain component implementations that wrap other libraries.

.. autoapisummary::

    lenskit.implicit
    lenskit.hpf
    lenskit.graphs

.. toctree::
    :caption: Add-Ons
    :hidden:

    lenskit/implicit/index
    lenskit/hpf/index
    lenskit/graphs/index


Implementation Helpers
~~~~~~~~~~~~~~~~~~~~~~

These modules provide various utilities and helpers used to implement LensKit,
and may be useful in building new models and components for LensKit.

.. autoapisummary::

    lenskit.lazy
    lenskit.logging
    lenskit.math
    lenskit.parallel
    lenskit.random
    lenskit.stats
    lenskit.testing
    lenskit.torch
    lenskit._accel

.. toctree::
    :hidden:
    :caption: Implementation Helpers

    lenskit/lazy/index
    lenskit/logging/index
    lenskit/math/index
    lenskit/parallel/index
    lenskit/random/index
    lenskit/stats/index
    lenskit/testing/index
    lenskit/torch/index

.. lenskit/_accel/index
