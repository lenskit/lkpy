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

Additional dataset construction support and possible implementations (e.g.
database-backed datasets) are coming, but this is the migration path for the
typical code patterns used in LensKit 0.14 and earlier.

.. tip::

    The :py:func:`~lenskit.data.load_movielens` function can now directly load
    MovieLens data from the ``.zip`` files distributed by GroupLens, without
    needing to extract them first.  It also automatically detects which version
    of the MovieLens data you are loading.

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
    ambiguous and promotes confusion about very different things.  Instead we
    have “pipelines” consisting of ”components”, some of which may be ”models”
    (for scoring, ranking, etc.).

Obtaining Recommendations
-------------------------

In previous LensKit versions, you would get recommendations by calling the
`recommend` method and providing the user ID, recommendation count, and
optionally the user's current historical ratings.

In LensKit 2025, you invoke the *pipeline* to obtain recommendations.  In a
standard recommendation pipeline, the recommendations are produced by a
component called ``recommender``; you can obtain them with:

.. code:: python

    recs = pipeline.run('recommender', query=user_id)

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
