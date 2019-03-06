Batch-Running Recommenders
==========================

.. highlight:: python
.. module:: lenskit.batch

The functions in :py:mod:`lenskit.batch` enable you to generate many recommendations or 
predictions at the same time, useful for evaluations and experiments.

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
