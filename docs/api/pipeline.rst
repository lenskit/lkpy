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
    ~lenskit.pipeline.PipelineBuilder
    ~lenskit.pipeline.PipelineState
    ~lenskit.pipeline.Node
    ~lenskit.pipeline.Lazy
    ~lenskit.pipeline.PipelineCache

Component Interface
-------------------

These are the interfaces and classes you need to reference when building new
LensKit components.

.. autosummary::
    :toctree: .
    :nosignatures:

    ~lenskit.pipeline.Component

Standard Pipelines
------------------

.. autosummary::
    :toctree: .
    :nosignatures:

    ~lenskit.pipeline.RecPipelineBuilder
    ~lenskit.pipeline.topn_pipeline
    ~lenskit.pipeline.predict_pipeline

Serialized Configurations
-------------------------

Most code will not need to directly use the configuration classes, but they define
the schema of pipeline configuration files.

.. autosummary::
    :toctree: .
    :nosignatures:

    lenskit.pipeline.config

Hook Interfaces
---------------

.. autosummary::
    :toctree: .
    :nosignatures:

    ~lenskit.pipeline.ComponentInputHook

Nodes
-----

The pipeline directly exposes its nodes, but most code will not need to use
them.

.. autosummary::
    :toctree: .
    :nosignatures:

    lenskit.pipeline.nodes
