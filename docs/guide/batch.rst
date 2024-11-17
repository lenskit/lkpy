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
-----------

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
    >>> split = sample_users(data, 150, SampleN(5))

Configure and train the model:

    >>> model = PopScorer()
    >>> pop_pipe = topn_pipeline(model, n=20)
    >>> pop_pipe.train(split.train)

Generate recommendations:

    >>> recs = recommend(pop_pipe, split.test.keys())
    >>> len(recs)
    150

And measure their results:

    >>> measure = RunAnalysis()
    >>> measure.add_metric(RBP())
    >>> scores = measure.compute(recs, split.test)
    >>> scores.summary()    # doctest: +ELLIPSIS
            mean    median     std
    RBP  0.07...    0.0...  0.1...


The :py:func:`predict` function works similarly, but for rating predictions;
instead of a simple list of user IDs, it takes a dictionary mapping user IDs to
lists of test items (as :py:class:`~lenskit.data.ItemList`).

General Batch Pipeline Runs
---------------------------

The :py:func:`recommend` and :py:func:`predict` functions are convenience
wrappers around a more general facility, the :py:class:`BatchPipelineRunner`.
