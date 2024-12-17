.. _pipeline:

Recommendation Pipelines
========================

.. py:currentmodule:: lenskit.pipeline

Since version :ref:`2025.1`, LensKit uses a flexible “pipeline” abstraction to
wire together different components such as candidate selectors, personalized
item scorers, and rankers to produce predictions, recommendations, or other
recommender system outputs.  This is a significant change from the LensKit 0.x
design of monolithic and composable components based on the Scikit-Learn API,
allowing new recommendation designs to be composed without writing new classes
just for the composition.  It also makes recommender definition code more
explicit by laying out the pipeline instead of burying composition logic in the
definitions of different composition classes.  The pipeline lives in the
:mod:`lenskit.pipeline` module, and the primary entry point is the
:class:`Pipeline` class.

If all you want to do is build a standard top-N recommendation pipeline from an
item scorer, see :func:`topn_pipeline`.  :class:`RecPipelineBuilder` provides
some more flexibility in configuring a recommendation pipeline with a standard
design, and you can always fully configure the pipeline yourself for maximum
flexibility.

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
pipeline can use some components (especially the scorer) trained from training
data, and other components that query a database or REST services for things
like user history and candidate set lookup.

.. admonition:: Acknowledgements
    :class: note

    The LensKit pipeline design is heavily inspired by the pipeline abstraction
    Karl Higley originally created for POPROX_ (available in the git history),
    as well as by Haystack_.

.. _Haystack: https://docs.haystack.deepset.ai/docs/pipelines
.. _POPROX: https://ccri-poprox.github.io/poprox-researcher-manual/reference/recommender/poprox_recommender.pipeline.html

.. _pipeline-construct:

Constructing Pipelines
~~~~~~~~~~~~~~~~~~~~~~

The simplest way to make a pipeline is to construct a ``topn_pipeline`` — the
following will create a top-*N* recommendation pipeline using implicit-feedback
matrix factorization:

.. code:: python

    als = ImplicitMF(50)
    pipe = topn_pipeline(als)

The :class:`RecPipelineBuilder` class provides a more flexible mechanism to
create standard recommendation pipelines; to implement the same pipeline with
that class, do:

.. code:: python

    als = ImplicitMF(50)
    builder = RecPipelineBuilder()
    builder.scorer(als)
    pipe = builder.build('ALS')

For maximum flexibility, you can directly construct and wire the pipeline
yourself; this is described in :ref:`standard-pipelines`.

After any of these methods, you can run the pipeline to produce recommendations
with:

.. code:: python

    user_recs = pipe.run(recommend, query=user_id)

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

You can also use :meth:`Pipeline.add_default` to specify default connections. For example,
you can specify a default for ``user``::

    pipe.add_default('user', user_history)

With this default in place, if a component has an input named ``user`` and that
input is not explicitly connected to a node, then the ``user_history`` node will
be used to supply its value.  Judicious use of defaults can reduce the amount of
code overhead needed to wire common pipelines.

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
    component that was added to the pipeline.

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

.. _pipeline-names:

Component Names
---------------

As noted above, each component (and pipeline input) has a *name* that is unique
across the pipeline.  For consistency and clarity, we recommend naming
components with a noun or kebab-case noun phrase that describes the component
itself, e.g.:

* ``recommender``
* ``reranker``
* ``scorer``
* ``history-lookup``
* ``item-embedder``

Component nodes can also have *aliases*, allowing them to be accessed by more
than one name. Use :meth:`Pipeline.alias` to define these aliases.

Various LensKit facilities recognize several standard component names used by
the standard pipeline builders, and we recommend you use them in your own
pipelines when applicable:

* ``scorer`` — compute (usually personalized) scores for items for a given user.
* ``ranker`` — compute a (ranked) list of recommendations for a user.  If you
  are configuring a pipeline with rerankers whose outputs are also rankings,
  this name should usually be used for the last such ranker, and downstream
  components (if any) transform that ranking into another layout; that way the
  evaluation tools will operate on the last such ranking.
* ``recommender`` — compute recommendations for a user.  This will often be an
  alias for ``ranker``, as in a top-*N* recommender, but may return other
  formats such as grids or unordered slates.
* ``rating-predictor`` — predict a user's ratings for the specified items.  When
  present, this may be an alias for ``scorer``, or it may be another component
  that fills in missing scores with a baseline prediction.

These component names replace the task-specific interfaces in pre-2025 LensKit;
a ``Recommender`` is now just a pipeline with ``recommender`` and/or ``ranker``
components.

.. _pipeline-serialization:

Pipeline Serialization
----------------------

Pipelines are defined by the following:

* The components and inputs (nodes)
* The component input connections (edges)
* The component configurations (see :class:`Configurable` and :class:`Component`)
* The components' learned parameters (see :class:`Trainable`)

LensKit supports serializing both pipeline descriptions (components,
connections, and configurations) and pipeline parameters.  There are
two ways to save a pipeline or part thereof:

1.  Pickle the entire pipeline.  This is easy, and saves everything in the
    pipeline; it has the usual downsides of pickling (arbitrary code execution,
    etc.). LensKit uses pickling to share pipelines with worker processes for
    parallel batch operations.
2.  Save the pipeline configuration with :meth:`Pipeline.save_config`.  This saves
    the components, their configurations, and their connections, but **not** any
    learned parameter data.  A new pipeline can be constructed from such a
    configuration can be reloaded with :meth:`Pipeline.from_config`.

.. comment::

    3.  Save the pipeline parameters with :meth:`Pipeline.save_params`.  This saves
        the learned parameters but **not** the configuration or connections.  The
        parameters can be reloaded into a compatible pipeline with
        :meth:`Pipeline.load_params`; a compatible pipeline can be created by
        running the same pipeline setup code or using a saved pipeline
        configuration.

    These can be mixed and matched: if you pickle an untrained pipeline, you can
    unpickle it and use :meth:`~Pipeline.load_params` to infuse it with parameters.

    Component implementations need to support the configuration and/or parameter
    values, as needed, in addition to functioning correctly with pickle (no specific
    logic is usually needed for this).

    LensKit knows how to safely save the following object types from
    :meth:`Trainable.get_params`:

    *   :class:`torch.Tensor` (dense, CSR, and COO tensors).
    *   :class:`numpy.ndarray`.
    *   :class:`scipy.sparse.csr_array`, :class:`~scipy.sparse.coo_array`,
        :class:`~scipy.sparse.csc_array`, and the corresponding ``*_matrix``
        versions.

    Other objects (including Pandas dataframes) are serialized by pickling, and the
    pipeline will emit a warning (or fail, if ``allow_pickle=False`` is passed to
    :meth:`~Pipeline.save_params`).

    .. note::

        The load/save parameter operations are modeled after PyTorch's
        :meth:`~torch.nn.Module.state_dict` and the needs of safetensors_.

    .. _safetensors: https://huggingface.co/docs/safetensors/

.. _standard-pipelines:

Standard Layouts
~~~~~~~~~~~~~~~~

The standard recommendation pipeline, produced by either of the approaches
described above in :ref:`pipeline-construct`, looks like this:

.. mermaid:: std-topn-pipeline.mmd
    :caption: Top-N recommendation pipeline.

The convenience methods are equivalent to the following pipeline code:

.. code:: python

    pipe = Pipeline()
    # define an input parameter for the user ID (the 'query')
    query = pipe.create_input('query', ID)
    # allow candidate items to be optionally specified
    items = pipe.create_input('items', ItemList, None)
    # look up a user's history in the training data
    history = pipe.add_component('history-lookup', LookupTrainingHistory(), query=query)
    # find candidates from the training data
    default_candidates = pipe.add_component(
        'candidate-selector',
        UnratedTrainingItemsCandidateSelector(),
        query=history,
    )
    # if the client provided items as a pipeline input, use those; otherwise
    # use the candidate selector we just configured.
    candidates = pipe.use_first_of('candidates', items, default_candidates)
    # score the candidate items using the specified scorer
    score = pipe.add_component('scorer', scorer, query=query, items=candidates)
    # rank the items by score
    recommend = pipe.add_component('ranker', TopNRanker(50), items=score)
    pipe.alias('recommender', recommend)


If we want to also emit rating predictions, with fallback to a baseline model to
predict ratings for items the primary scorer cannot score (e.g. they are not in
an item neighborhood), we use the following pipeline (created by
:class:`RecPipelineBuilder` when rating prediction is enabled):

.. mermaid:: std-pred-pipeline.mmd
    :caption: Pipeline for top-N recommendation and rating prediction, with predictions falling back to a baseline scorer.


Component Interface
~~~~~~~~~~~~~~~~~~~

Pipeline components are callable objects that can optionally provide
configuration, training, and serialization capabilities.  In the simplest case,
a component that requires no training or configuration can simply be a Python
function; most components will extend the :class:`Component` base class to
expose configuration capabilities, and implement the :class:`Trainable` protocol
if they contain a model that needs to be trained.

Components also must be pickleable, as LensKit uses pickling for shared memory
parallelism in its batch-inference code.
