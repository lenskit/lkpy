Operation Functions
===================

.. py:module:: lenskit.operations

The :py:mod:`lenskit.operations` module defines convenience functions for
various recommender operations, simplifying the calls to the underlying
pipeline.  Each of these functions takes a pipeline, along with some parameters
(e.g. the user ID or query), and runs the pipeline with those options.

These functions are re-exported from the top level ``lenskit`` package, so you
can directly import them:

.. code:: python

    from lenskit import recommend, score

Recommending
~~~~~~~~~~~~

This function is the primary recommendation function to obtain a list of
recommended items.

.. autofunction:: recommend

Scoring and Predicting
~~~~~~~~~~~~~~~~~~~~~~

These functions score individual items with respect to a query (e.g. a user ID
or history); they differ only in their default component.

.. autofunction:: score
.. autofunction:: predict
