Batch-Running Recommenders
==========================

.. highlight:: python
.. module:: lenskit.batch

The functions in :py:mod:`lenskit.batch` enable you to generate many recommendations or 
predictions at the same time, useful for evaluations and experiments.

The batch functions can parallelize over users with the optional ``n_jobs`` parameter, or
the ``LK_NUM_PROCS`` environment variable.

.. note::
    Scripts calling the batch recommendation or prediction facilites must be *protected*;
    that is, they should not directly perform their work when run, but should define functions
    and call a ``main`` function when run as a script, with a block like this at the end of the
    file::

        def main():
            # do the actual work

        if __name__ == '__main__':
            main()
    
    If you are using the batch functions from a Jupyter notbook, you should be fine - the
    Jupyter programs are appropriately protected.

Recommendation
~~~~~~~~~~~~~~

.. autofunction:: recommend

Rating Prediction
~~~~~~~~~~~~~~~~~

.. autofunction:: predict

Scripting Evaluation
~~~~~~~~~~~~~~~~~~~~

The :py:class:`MultiEval` class is useful to build scripts that evaluate multiple algorithms
or algorithm variants, simultaneously, across multiple data sets. It can extract parameters
from algorithms and include them in the output, useful for hyperparameter search.

For example::

    from lenskit.batch import MultiEval
    from lenskit.crossfold import partition_users, SampleN
    from lenskit.algorithms import basic, als
    from lenskit.datasets import MovieLens
    from lenskit import topn
    import pandas as pd

    ml = MovieLens('ml-latest-small')

    eval = MultiEval('my-eval', recommend=20)
    eval.add_datasets(partition_users(ml.ratings, 5, SampleN(5)), name='ML-Small')
    eval.add_algorithms(basic.Popular(), name='Pop')
    eval.add_algorithms([als.BiasedMF(f) for f in [20, 30, 40, 50]],
                        attrs=['features'], name='ALS')
    eval.run()

The ``my-eval/runs.csv`` file will then contain the results of running these 
algorithms on this data set.  A more complete example is available in the
`MultiEval notebook`_.

.. _MultiEval notebook: https://nbviewer.jupyter.org/github/lenskit/lkpy/blob/master/examples/MultiEval.ipynb

.. autoclass:: MultiEval
    :members:
