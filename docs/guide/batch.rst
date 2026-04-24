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
    >>> from lenskit.metrics import MeasurementCollector, RBP

Load and split some data:

    >>> data = load_movielens('data/ml-100k.zip')
    >>> split = sample_users(data, 150, SampleN(5, rng=1024), rng=42)

Configure and train the model:

    >>> model = PopScorer()
    >>> pop_pipe = topn_pipeline(model, n=20)
    >>> pop_pipe.train(split.train)

Generate recommendations:

    >>> recs = recommend(pop_pipe, split.test.keys())
    >>> recs.to_df()
              user_id  item_id     rank  score
    0 ...                             1 ...
    ...
    [3000 rows x 4 columns]

And measure their results:

    >>> collect = MeasurementCollector()
    >>> collect.add_metric(RBP())
    >>> collect.add_collection_measurements(recs, split.test)
    >>> collect.summary_metrics()    # doctest: +ELLIPSIS
    {... 'RBP.mean': 0.06..., ...}


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

The batch inference functions and methods (:func:`~lenskit.batch.recommend`,
:meth:`~lenskit.batch.BatchPipelineRunner.run`, etc.) batch recommendation
requests in the following formats:

* An iterable of identifiers, which are taken to be user IDs and used as both the
  :attr:`~lenskit.data.RecQuery.user_id` and the
  :attr:`~lenskit.data.RecQuery.query_id` to make otherwise-empty
  :attr:`queries <lenskit.data.RecQuery>`.

* An iterable of :attr:`~lenskit.data.RecQuery` objects, specifying the
  recommendation queries.

* An iterable of dictionaries, each of which conforms to
  :class:`~lenskit.batch.BatchRecRequest`.

* An :class:`~lenskit.data.ItemListCollection`, which is translated to a
  collection of recommendation requests with :class:`TestRequestAdapter`.  This
  translation extracts user and query IDs from the item list keys, and uses the
  item lists themselves as test items for rating prediction (and uses default
  candidate sets for recommendation).

.. versionchanged:: 2026.1

    Restricted the batch inputs to the types above, in order to make input
    clearer and more explicit.
