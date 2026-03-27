.. _batch:

Batch-Running Pipelines
=======================

.. py:currentmodule:: lenskit.batch

.. highlight:: python

Offline recommendation experiments require *batch-running* a pipeline over a set
of test users, sessions, or other recommendation requests.  LensKit supports this
through the facilities in the :py:mod:`lenskit.batch` module.

By default, the batch facilities operate in parallel over the test users; this
can be controlled by environment variables (see :ref:`parallel-config`) or
through an ``n_jobs`` keyword argument to the various functions and classes.

.. admonition:: Import Protection
    :class: important

    Scripts using batch pipeline operations must be *protected*; see
    :ref:`parallel-protecting`.

Simple Runs
~~~~~~~~~~~

If you have a pipeline and want to simply generate recommendations for a batch
of test users, you can do this with the :py:func:`recommend` function.

For an example, let's start with importing things to run a quick batch:

    >>> from lenskit.basic import PopScorer
    >>> from lenskit.pipeline import topn_pipeline
    >>> from lenskit.batch import recommend
    >>> from lenskit.data import load_movielens
    >>> from lenskit.splitting import sample_users, SampleN
    >>> from lenskit.metrics import RunAnalysis, RBP

Load and split some data:

    >>> data = load_movielens('data/ml-100k.zip')
    >>> split = sample_users(data, 150, SampleN(5, rng=1024), rng=42)

Configure and train the model:

    >>> model = PopScorer()
    >>> pop_pipe = topn_pipeline(model, n=20)
    >>> pop_pipe.train(split.train)

Generate recommendations:

    >>> recs = recommend(pop_pipe, split.test.keys(), n_jobs=1)
    >>> recs.to_df()
              user_id  item_id     score  rank
    0 ...                                    1
    ...
    [3000 rows x 4 columns]

And measure their results:

    >>> ra = RunAnalysis()
    >>> ra.add_metric(RBP())
    >>> scores = ra.measure(recs, split.test)
    >>> scores.list_summary()    # doctest: +ELLIPSIS
              mean    median     std
    metric
    RBP    0.06...   0.02... 0.07...


The :py:func:`predict` function works similarly, but for rating predictions.
Instead of a simple list of user IDs, it takes a dictionary mapping user IDs to
lists of test items (as :py:class:`~lenskit.data.ItemList`).

General Batch Pipeline Runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:func:`recommend` and :py:func:`predict` functions are convenience
wrappers around a more general facility, the :py:class:`BatchPipelineRunner`.

.. _batch-queries:

Batch Queries
~~~~~~~~~~~~~

.. py:currentmodule:: lenskit.data

The batch inference functions and methods (:func:`~lenskit.batch.recommend`,
:meth:`~lenskit.batch.BatchPipelineRunner.run`, etc.) accept multiple types of
input to specify the set of users or test items.

* An iterable (e.g. list) of recommendation queries (as :class:`RecQuery`
  objects).  The queries must have at least one of :attr:`RecQuery.query_id` and
  :attr:`RecQuery.user_id` set, so that the output can be properly indexed.
  Queries should all have the identification method (i.e., all queries have a
  ``query_id``, or all queries have only a ``user_id``).

* An iterable of 2-element ``(query, items)`` tuples.  The query is a
  :class:`RecQuery` as in the previous method, and the items is an :class:`ItemList`
  containing the candidate items (for recommendation) or the items to score (for
  prediction and scoring).  This is the most general form of input.

* An iterable (e.g. list) of user IDs.  These are passed as
  :attr:`RecQuery.user_id`, and the resulting outputs are mapped to  ID.

* An :class:`ItemListCollection`.  At least one field of the collection key
  should be ``user_id``, and these user IDs are used as the query user IDs. The
  item lists themselves are used as in the tuple method above. Results are
  indexed by the entire key.

* A mapping (dictionary) of IDs to item lists.  This behaves like the item
  list collection; the IDs are taken to be user IDs.

* A :class:`pandas.DataFrame`, which is converted to an item list collection.

.. deprecated:: 2025.6

    Mappings and data frames are deprecated in favor of other input types.
