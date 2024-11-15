.. _batch:

Batch-Running Pipelines
=======================

.. py:currentmodule:: lenskit.batch

.. highlight:: python

The functions and :py:class:`BatchPipelineRunner` class in
:py:mod:`lenskit.batch` enable you to generate many recommendations or
predictions at the same time, useful for evaluations and experiments.

.. admonition:: Import Protection
    :class: important

    Scripts using batch pipeline operations must be *protected*; that is, they
    should not directly perform their work when run, but should define functions
    and call a ``main`` function when run as a script, with a block like this at
    the end of the file::

        def main():
            # do the actual work

        if __name__ == '__main__':
            main()

    If you are using the batch functions from a Jupyter notebook, you should be fine - the
    Jupyter programs are appropriately protected.

Quick-and-Dirty Runs
--------------------

If you have a pipeline and want to simply generate recommendations for a batch
of test users, you can do this with the :py:func:`recomend` function (or
:py:func:`predict` for rating predictions).  For example:

>>> from lenskit.basic import PopScorer
>>> from lenskit.pipeline import topn_pipeline
>>> from lenskit.batch import recommend
>>> from lenskit.data import load_movielens
>>> from lenskit.splitting import sample_users, SampleN
>>> from lenskit.metrics import RunAnalysis, RBP
>>> data = load_movielens('data/ml-100k.zip')
>>> split = sample_users(data, 150, SampleN(5))
>>> model = PopScorer()
>>> pop_pipe = topn_pipeline(model, n=20)
>>> pop_pipe.train(split.train)
>>> recs = recommend(pop_pipe, split.test.keys())
>>> measure = RunAnalysis()
>>> measure.add_metric(RBP())
>>> scores = measure.compute(recs, split.test)
>>> scores.summary()
       mean  median       std
RBP  0.0358     0.0  0.1...
