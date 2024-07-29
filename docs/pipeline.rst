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

If all you want to do is build a standard top-N recommendation pipeline from an
item scorer, see :func:`topn_pipeline`; this is the equivalent to
``Recommender.adapt`` in the old LensKit API.  If you want more flexibility, you
can write out the pipeline configuration yourself; the equivalent to
``topn_pipeline(scorer)`` is:

.. code:: python

    pipe = Pipeline()
    # define an input parameter for the user ID
    user = pipe.create_input('user', EntityId)
    # allow candidate items to be optionally specified
    items = pipe.create_input('items', list[EntityId], None)
    # look up a user's history in the training data
    history = pipe.add_component('lookup-user', LookupTrainingHistory(), user=user)
    # find candidates from the training data
    lookup_candidates = pipe.add_component(
        'select-candidates',
        UnratedTrainingItemsCandidateSelector(),
        user=history,
    )
    # if the client provided items as a pipeline input, use those; otherwise
    # use the candidate selector we just configured.
    candidates = pipe.use_first_of('candidates', items, lookup_candidates)
    # score the candidate items using the specified scorer
    scores = pipe.add_component('score', scorer, user=user, items=candidates)
    # rank the items by score
    recs = pipe.add_component('recommend', TopNRanker(50), items=scores)

.. todo::
    Redo some of those types with user & item data, etc.

.. todo::
    Provide utility functions to make more common wiring operations easy so there
    is middle ground between “give me a standard pipeline” and “make me do everything
    myself”.

Pipeline components are not limited to looking things up from training data —
they can query databases, load files, and any other operations.  A runtime
pipeline can use some (especially the scorer) trained from training data, and
other components that query a database or REST services for things like user
history and candidate set lookup.

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
