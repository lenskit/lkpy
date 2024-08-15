Recommendation Pipelines
========================

.. py:module:: lenskit.pipeline

This page documents the LensKit pipeline API, exposed in the
:mod:`lenskit.pipeline` module

Pipeline Classes
----------------

.. autosummary::
    :toctree: .
    :nosignatures:

    Pipeline
    Node

Component Interface
-------------------

These are the interfaces and classes you need to reference when building new
LensKit components.

.. autosummary::
    :toctree: .
    :nosignatures:

    ConfigurableComponent
    TrainableComponent
    AutoConfig

Standard Pipelines
------------------

.. autosummary::
    :toctree: .
    :nosignatures:

    topn_pipeline
