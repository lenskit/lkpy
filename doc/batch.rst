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

Isolated Training
~~~~~~~~~~~~~~~~~

This function isn't a batch function per se, as it doesn't perform multiple operations, but it
is primarily useful with batch operations.  The :py:func:`train_isolated` function trains an
algorithm in a subprocess, so all temporary resources are released by virtue of the training
process exiting.  It returns a shared memory serialization of the trained model, which can
be passed directly to :py:func:`recommend` or :py:func:`predict` in lieu of an algorithm object,
to reduce the total memory consumption.

Example usage::

    algo = BiasedMF(50)
    algo = Recommender.adapt(algo)
    algo = batch.train_isolated(algo, train_ratings)
    preds = batch.predict(algo, test_ratings)

.. autofunction:: train_isolated


Scripting Evaluation
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiEval
    :members:
