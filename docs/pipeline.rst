Recommendation Pipelines
========================

.. module:: lenskit.pipeline

Since version :ref:`2024.1`, LensKit uses a flexible “pipeline” abstraction to
wire together different components such as candidate selectors, personalized
item scorers, and rankers to produce predictions, recommendations, or other
recommender system outputs.  This is a significant change from the LensKit 0.x
design of monolithic and composable components based on the Scikit-Learn API,
allowing new recommendation designs to be composed without writing new classes
just for the composition.  It also makes recommender definition code more explicit
by laying out the pipeline instead of burying composition logic in the definitions
of different composition classes.

If all you want to do is build a standard top-N recommendation pipeline from an item
scorer, see :func:`topn_pipeline`.

The LensKit pipeline design is heavily inspired by Haystack_ and by the pipeline
abstraction Karl Higley created for POPROX_.

.. _Haystack: https://docs.haystack.deepset.ai/docs/pipelines
.. _POPROX: https://ccri-poprox.github.io/poprox-researcher-manual/reference/recommender/poprox_recommender.pipeline.html

Common Pipelines
~~~~~~~~~~~~~~~~

These functions make it easy to create common pipeline designs.

.. autofunction:: topn_pipeline


Pipeline Class
~~~~~~~~~~~~~~

.. autoclass:: Pipeline

Pipeline Nodes
~~~~~~~~~~~~~~

.. autoclass:: Node

Component Interface
~~~~~~~~~~~~~~~~~~~

Pipeline components are callable objects that can optionally provide training
and serialization capabilities.  In the simplest case, a component that requires
no training or configuration can simply be a Python function; more sophisticated
components can implement the :class:`TrainableComponent` and/or
:class:`ConfigurableComponent` protocols to support flexible model training and
pipeline serialization.

.. note::

    The component interfaces are simply protocol definitions (defined using
    :class:`typing.Protocol` with :func:`~typing.runtime_checkable`), so
    implementations can directly implement the specified methods and do not need
    to explicitly inherit from the protocol classes, although they are free to
    do so.

.. todo::

    Is it clear to write these capabilities as separate protocols, or would it be
    better to write a single ``Component`` :class:`~abc.ABC`?

.. autoclass:: Component

.. autoclass:: ConfigurableComponent

.. autoclass:: TrainableComponent
