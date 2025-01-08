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

    ~lenskit.pipeline.Pipeline
    ~lenskit.pipeline.PipelineState
    ~lenskit.pipeline.Node
    ~lenskit.pipeline.Lazy

Component Interface
-------------------

These are the interfaces and classes you need to reference when building new
LensKit components.

.. autosummary::
    :toctree: .
    :nosignatures:

    ~lenskit.pipeline.Component
    ~lenskit.pipeline.Trainable

Standard Pipelines
------------------

.. autosummary::
    :toctree: .
    :nosignatures:

    ~lenskit.pipeline.RecPipelineBuilder
    ~lenskit.pipeline.topn_pipeline

Serialized Configurations
-------------------------

.. autosummary::
    :toctree: .
    :nosignatures:

    ~lenskit.pipeline.PipelineConfig
