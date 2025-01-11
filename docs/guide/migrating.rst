.. _migrating:

Migrating from LensKit 0.x
==========================

.. versionchanged:: 2025.1

    Everything.

LensKit 2025.1 brings the largest changes to LensKit's APIs and operation since
moving to Python.  These changes are a result of years of experience using and
teaching the current Python APIs, talking with users (and non-users), and
understanding limitations of the previous design in the modern recommendation
and machine learning landscape, and “pathway limitations” — things that were
technically possible, but the design did not make it clear how to achieve them.

The recommendation models and evaluation metrics retain their behavior from
previous versions of LensKit, but their interfaces and the surrounding
infrastructure are significantly different.

The new changes will also enable new functionality that is planned for upcoming
versions of LensKit, such as first-class content-based and hybrid models, and
support for additional recommendation tasks such as session- or context-based
recommendation.

Change Highlights
-----------------

The major changes that client or research code will need to adapt to largely
fall into the following categories:

- **Data structures** — LensKit now uses recommender-oriented data abstractions,
  such as :class:`~lenskit.data.Dataset` (see :ref:`datasets`) and
  :class:`~lenskit.data.ItemList`, instead of storing everything directly in
  Pandas data frames.  All data structures support round-tripping with data
  frames (and often other structures); the default user and item column names
  have changed (to ``user_id`` and ``item_id``).

- **Computational infrastructure** — LensKit models now mostly use PyTorch
  instead of Numba-accelerated NumPy code.  This speeds forward compatibility,
  as Numba was often a release bottleneck.  There are no limits on what can be
  used in models, however — TensorFlow and Jax models should work just fine in
  LensKit.

  LensKit also uses SPEC 7 instead of SeedBank for configuring random number
  generation support (see :ref:`rng`).

- **Pipelines** — LensKit recommendation is now built around *pipelines* (see
  :ref:`pipeline`) consisting of individual *components*, such as scoring models
  and rankers.

- **Queries** — requests for recommendations are now encapsulated in a “query”
  (implemented by :py:class:`~lenskit.data.RecQuery`) containing the different
  features to identify and describe a recommendation request, such as a user
  identifier, historical ratings, or session information (see :ref:`queries`).

- **Reorganization** — the ``lenskit.algorithms`` package is gone, and
  recommendation components are directly in packages such as
  :py:mod:`lenskit.basic` and :py:mod:`lenskit.als`.

- **Evaluation** — the new :py:class:`~lenskit.metrics.RunAnalysis` class
  provides unified handling of both prediction and recommendation metrics,
  summary statistics, and better defaults and handling for missing data.  It
  also directly supports “global” metrics that are computed over an entire run
  instead of one list at a time.

Loading Data
------------

New code should use :py:func:`lenskit.data.from_interactions_df` to convert a Pandas
data frame into a :py:func:`~lenskit.data.Dataset`, or one of the standard loaders
such as :py:func:`lenskit.data.load_movielens`.

While most LensKit data frame code still recognizes the legacy ``user`` and
``item`` columns from LensKit 0.14 and earlier, data frames of LensKit data
should use the column names ``user_id`` and ``item_id`` instead, to
unambiguously distinguish them from user and item numbers.

Additional dataset construction support and possible implementations (e.g.
database-backed datasets) are coming, but this is the migration path for the
typical code patterns used in LensKit 0.14 and earlier.

.. tip::

    The :py:func:`~lenskit.data.load_movielens` function can now directly load
    MovieLens data from the ``.zip`` files distributed by GroupLens, without
    needing to extract them first.  It also automatically detects which version
    of the MovieLens data you are loading.

Data Structures
---------------

Where older versions of LensKit used Pandas data frames and series as the
primary data structures for interfacing with components, LensKit 2025 introduces
new data abstractions specifically for handling recommender data, but that support
conversion to and from data frames.  The core ones are:

- :class:`~lenskit.data.ItemList` represents a list of items, optionally with
  scores or other fields (e.g. ratings).  Item lists can convert between item
  IDs and item numbers, using a vocabulary, and can be converted to and from
  Pandas data frames.  Their fields (including the item numbers) can also be
  retrieved in multiple formats, including NumPy arrays (the default), Pandas
  :class:`~pandas.Series`, and PyTorch tensors.  Format conversions are
  zero-copy whenever possible.

- :class:`~lenskit.data.Vocabulary` represents a collection of item or user IDs
  (or other ID-like things, such as tags), and supports bidirectional mapping
  between such IDs and contiguous 0-based indices (numbers) for indexing into
  arrays and matrices.  This was not used as a part of an API in LensKit before,
  but was implemented internally by many components using the Pandas
  :class:`~pandas.Index` data structure.  Vocabularies centralize that logic
  (and use :class:`~pandas.Index` under the hood), so that we don't duplicate it
  so much across the codebase and to enable multiple models trained on the same
  data to share the same index.  If you are implementing a model component that
  needs to store vectors or matrices of user or item data, consider using the
  vocabulary to associate those with user and item IDs.

- :class:`~lenskit.data.ItemListCollection` represents a collection of item
  lists indexed by keys, such as the test items for users a test data split, or
  the recommendation lists for users in an experiment.  It supports conversion
  to and from Pandas data frames.  Future releases will support additional
  formats, such as DuckDB.

Motivation
..........

These data structures, and the data set abstraction, are something of a
departure from one of the design principles originally set out for LensKit for
Python :cite:p:`lkpy`; specifically, to use standard data structures for
interchange between components.

There are three primary reason for this change:

* While Pandas data frames and series are widely used and supported by many
  libraries, they are not self-documenting: a Python method returning a
  :class:`~pandas.DataFrame` is not enough to know what columns in that data
  frame.  Things are further complicated with Pandas indexes, requiring
  elaborate discussions of exact data frame and series layouts in the
  documentation.  This also sometimes resulted in bugs with incorrect layouts,
  particularly if an index was incorrectly configured.  Dedicated abstractions
  are more self-documenting, particularly in modern Python with type annotations
  and good IDE support.

* Many libraries work directly with arrays and sparse matrices instead of Pandas
  data structures, requiring data conversion and translation that is often
  repeated in different model components.  First-class support for multiple data
  formats in a single abstraction reduces the work needed to implement a model
  with PyTorch, Scikit-Learn, or any other library.

* When chaining together multiple components, data always needed to be converted
  to and from Pandas at the component interface boundary.  This meant that two
  components both using PyTorch needed to convert to Pandas (possibly moving
  from GPU to CPU) at the interface, and then convert back to PyTorch.  A
  unified interface with lazy, zero-copy conversion means that two components
  using the same compute support do not need to convert data in order to
  interface, while still supporting composition with arbitrary components using
  different compute layers.

Since the new data structures, particularly :class:`ItemList`, are thin
abstractions on top of arrays, these are hopefully still as easy (or easier) to
use and integrate, and provide much easier support for implementing new
components with your choice of support libraries.

Configuring Recommenders
------------------------

In LensKit 0.3 through 0.14, you configured a recommender by instantiating an
*algorithm*, and then calling ``Recommender.adapt`` to make sure it implemented
the ``Recommender`` interface.

LensKit 2025 introduces the *pipeline* design; you configure the core
recommendation model in a very similar way (constructor arguments), and pass it
to :py:func:`~lenskit.pipeline.topn_pipeline` instead of ``Recommender.adapt``.
The resulting pipeline object can be directly used by the batch inference
facilities.

The model and pipeline training method is now named ``train``, so after creating
the pipeline, you will call :py:meth:`~lenskit.pipeline.Pipeline.train`::

    pipe.train(dataset)

See :ref:`pipeline` for more details on pipelines and how you can reconfigure
them for very different ways of turning scoring models into full recommenders.

.. note::

    Since 2025, we no longer use the term “algorithm” in LensKit, as it is
    ambiguous and promotes confusion about very different things.  Instead, we
    have “pipelines” consisting of ”components”, some of which may be ”models”
    (for scoring, ranking, etc.).

Configuration Components
........................

Individual components now use Pydantic_ models to represent their configuration
(e.g. hyperparameters).  This is to reduce redundancy, improve documentation,
enable consistent serialization, and validate parameter values in a consistent
and automated fashion.  See :ref:`component-config` for details.

Obtaining Recommendations
-------------------------

In previous LensKit versions, you would get recommendations by calling the
`recommend` method and providing the user ID, recommendation count, and
optionally the user's current historical ratings.

In LensKit 2025, you invoke the *pipeline* to obtain recommendations.  In a
standard recommendation pipeline, the recommendations are produced by a
component called ``recommender``; you can obtain them with the
:func:`~lenskit.recommend` function:

.. code:: python

    from lenskit import recommend
    recs = recommend(pipe, user_id)

This method returns an :py:class:`~lenskit.data.ItemList` containing the
recommended items. You can optionally specify candidate items with an ``items=``
parameter to ``run`` (it takes an :py:class:`~lenskit.data.ItemList`), or a list
length with ``n=`` (you can also bake a default list length into the pipeline
when you call :py:func:`~lenskit.pipeline.topn_pipeline`).

.. important::

    The input specifying the user identifier is now called a ``query``, in order
    to support recommendation tasks beyond simple user-based recommendation such
    as context-based or session-based recommendation.

.. note::

    We are considering adding a more ergonomic interface to obtain
    recommendations from pipelines.

Batch Inference
---------------

The :py:func:`~lenskit.batch.recommend` and :py:func:`~lenskit.batch.predict`
functions still exist, and now work on pipelines instead of “algorithms”. They
no longer return data frames; instead, they return an
:py:class:`~lenskit.data.ItemListCollection` containing the item lists produced
by the recommender or predictor / scorer components.

You can also use the more flexible
:py:class:`~lenskit.batch.BatchPipelineRunner` to do things like extract
multiple component outputs for each test user (e.g. both rating predictions and
top-*N* recommendations, or rankings before and after a reranking stage).

All batch inference interfaces support parallel processing over users, and the
same parallel configuration (see :ref:`parallelism`).  The resulting item list
collections can be converted to data frames
(:py:meth:`~lenskit.data.ItemListCollection.to_df`) to be saved in any format
supported by Pandas; future LensKit versions will add support for directly
storing them in other formats such as DuckDB, and loading them from such
formats.

Evaluating Recommendations
--------------------------

The evaluation logic has seen significant updates and improvements and API
changes. The :py:mod:`lenskit.splitting` module contains various facilities for
data splitting, including equivalents of the splitting strategies that used to
live in ``lenskit.crossfold``; see :ref:`splitting` for details on data
splitting.  These functions now operate on data sets and return item list
collections instead of data frames.

To measure recommendations, use the various metrics in
:py:mod:`lenskit.metrics`, and the :py:mod:`lenskit.metrics.RunAnalysis` class
provides support for analyzing *runs* (sequences of recommendation lists
produced by an algorithm in an experimental condition). It handles both ranking
and prediction accuracy metrics in a single analysis interface, and also
supports both listwise and global metrics (e.g. exposure metrics).  We will be
quickly building out additional metrics that take advantage of this
functionality.  See :ref:`evaluation` for details on metrics and analysis.

:py:mod:`lenskit.metrics.RunAnalysis` replaces the old ``RecListAnalysis``, and
provides better defaults (e.g. how users without recommendations are handled).
