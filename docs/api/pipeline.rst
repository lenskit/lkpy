Pipeline API
============

.. py:module:: lenskit.pipeline

This page documents the LensKit pipeline API, exposed in the
:mod:`lenskit.pipeline` module.

Pipeline Classes
----------------

.. autosummary::
    :toctree: .
    :nosignatures:
    :caption: Data Sets

    Pipeline
    PipelineBuilder
    PipelineState
    Node
    Lazy

Component Interface
-------------------

These are the interfaces and classes you need to reference when building new
LensKit components.

.. autosummary::
    :toctree: .
    :nosignatures:

    Component

Standard Pipelines
------------------

.. autosummary::
    :toctree: .
    :nosignatures:

    RecPipelineBuilder
    topn_pipeline
    predict_pipeline

Serialized Configurations
-------------------------

.. autosummary::
    :toctree: .
    :nosignatures:

    PipelineConfig
