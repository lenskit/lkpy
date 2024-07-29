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
    score = pipe.add_component('score', scorer, user=user, items=candidates)
    # rank the items by score
    recommend = pipe.add_component('recommend', TopNRanker(50), items=score)

You can then run this pipeline to produce recommendations with:

.. code:: python

    user_recs = pipe.run(recommend, user=user_id)

.. todo::
    Redo some of those types with user & item data, etc.

.. todo::
    Provide utility functions to make more common wiring operations easy so there
    is middle ground between “give me a standard pipeline” and “make me do everything
    myself”.

.. todo::
    Rethink the “keyword inputs only” constraint in view of the limitation it
    places on fallback or other compositional components — it's hard to specify
    a component that implements fallback logic for an arbitrary number of
    inputs.

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

.. _pipeline-model:

Pipeline Model
~~~~~~~~~~~~~~

A pipeline has a couple key concepts:

* An **input** is data that needs to be provided to the pipeline when it is run,
  such as the user to generate recommendations for.  Inputs have specified data
  types, and it is an error to provide an input value of an unexpected type.
* A **component** processes input data and produces an output.  It can be either
  a Python function or object (anything that implements the :class:`Component`
  protocol) that takes inputs as keyword arguments and returns an output.

These are arranged in a directed acyclic graph, consisting of:

* **Nodes** (represented by :class:`Node`), which correspond to either *inputs*
  or *components*.
* **Connections** from one node's input to another node's data (or to a fixed
  data value).  This is how the pipeline knows which components depend on other
  components and how to provide each component with the inputs it requires; see
  :ref:`pipeline-connections` for details.

Each node has a name that can be used to look up the node with
:meth:`Pipeline.node` and appears in serialization and logging situations. Names
must be unique within a pipeline.

.. _pipeline-connections:

Connections
-----------

Components declare their inputs as keyword arguments on their call signatures
(either the function call signature, if it is a bare function, or the
``__call__`` method if it is implemented by a class).  In a pipeline, these
inputs can be connected to a source, which the pipeline will use to obtain a
value for that parameter when running the pipeline.  Inputs can be connected to
the following types:

* A :class:`Node`, in which case the input will be provided from the
  corresponding pipeline input or component return value.  Nodes are
  returned by :meth:`create_input` or :meth:`add_component`, and can be
  looked up after creation with :meth:`node`.
* A Python object, in which case that value will be provided directly to
  the component input argument.

These input connections are specified via keyword arguments to the
:meth:`Pipeline.add_component` or :meth:`Pipeline.connect` methods — specify the
component's input name(s) and the node or data to which each input should be
wired.

.. note::

    You cannot directly wire an input another component using only that
    component's name; if you only have a name, pass it to :meth:`node`
    to obtain the node.  This is because it would be impossible to
    distinguish between a string component name and a string data value.

.. note::

    You do not usually need to call this method directly; when possible,
    provide the wirings when calling :meth:`add_component`.

.. _pipeline-execution:

Execution
---------

Once configured, a pipeline can be run with :meth:`Pipeline.run`.  This
method takes two types of inputs:

*   Positional arguments specifying the node(s) to run and whose results should
    be returned.  This is to allow partial runs of pipelines (e.g. to only score
    items without ranking them), and to allow multiple return values to be
    obtained (e.g. initial item scores and final rankings, which may have
    altered scores).

    If no components are specified, it is the same as specifying the last
    component added to the pipeline.

*   Keyword arguments specifying the values for the pipeline's inputs, as defined by
    calls to :meth:`create_input`.

Pipeline execution logically proceeds in the following steps:

1.  Determine the full list of pipeline components that need to be run
    in order to run the specified components.
2.  Run those components in order, taking their inputs from pipeline
    inputs or previous components as specified by the pipeline
    connections and defaults.
3.  Return the values of the specified components.  If a single
    component is specified, its value is returned directly; if two or
    more components are specified, their values are returned in a tuple.

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
