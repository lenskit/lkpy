Batch-Running Recommenders
==========================

.. highlight:: python
.. module:: lenskit.batch

The functions in :py:mod:`lenskit.batch` enable you to generate many recommendations or 
predictions at the same time, useful for evaluations and experiments.

The batch functions can parallelize over users.

.. note::
    Scripts calling the batch recommendation or prediction facilites must be *protected*;
    that is, they should not directly perform their work when run, but should define functions
    and call a ``main`` function when run as a script, with a block like this at the end of the
    file::

        def main():
            # do the actual work

        if __name__ == '__main__':
            main()
    
    This is to ensure compatibility with the parallel processing code, including in future
    LensKit versions.  If you are using the batch functions from a Jupyter notbook, you
    should be fine - the Jupyter programs are appropriately protected.

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

.. include:: MultiEvalExample.rst

Multi-Eval Class Reference
--------------------------

.. autoclass:: MultiEval
    :members:
